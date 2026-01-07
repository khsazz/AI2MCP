"""MCP Server with SSE transport for ROS 2 integration.

This module implements the main MCP server that exposes robot capabilities
as Tools (actions) and Resources (state) over Server-Sent Events.

Optionally integrates LeRobot datasets for offline imitation learning experiments.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import Response

from mcp_ros2_bridge.ros_node import ROS2Bridge
from mcp_ros2_bridge.tools import register_tools
from mcp_ros2_bridge.resources import register_resources

logger = structlog.get_logger()

# LeRobot integration flag - set via environment or constructor
LEROBOT_ENABLED = os.environ.get("LEROBOT_ENABLED", "false").lower() == "true"
LEROBOT_REPO_ID = os.environ.get("LEROBOT_REPO_ID", "lerobot/aloha_static_coffee")
LEROBOT_MODEL_PATH = os.environ.get("LEROBOT_MODEL_PATH", "")


class MCPRos2Server:
    """MCP Server bridging AI models to ROS 2.

    Supports two modes:
    - ROS 2 mode: Connect to live robot via ROS 2 bridge
    - LeRobot mode: Offline imitation learning with LeRobot datasets
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        enable_lerobot: bool = False,
        lerobot_repo_id: str | None = None,
        lerobot_model_path: str | None = None,
    ):
        self.host = host
        self.port = port
        self.enable_lerobot = enable_lerobot or LEROBOT_ENABLED
        self.lerobot_repo_id = lerobot_repo_id or LEROBOT_REPO_ID
        self.lerobot_model_path = lerobot_model_path or LEROBOT_MODEL_PATH or None

        self.mcp_server = Server("mcp-ros2-bridge")
        self.ros_bridge: ROS2Bridge | None = None
        self.sse_transport: SseServerTransport | None = None

        # LeRobot components (initialized lazily)
        self._lerobot_resource_manager = None
        self._lerobot_tools_manager = None

    async def initialize(self) -> None:
        """Initialize ROS 2 bridge and register MCP handlers."""
        logger.info("Initializing MCP-ROS2 Bridge", host=self.host, port=self.port)

        # Initialize ROS 2 bridge
        self.ros_bridge = ROS2Bridge()
        await self.ros_bridge.initialize()

        # Optionally initialize LeRobot integration first (for tool registration)
        if self.enable_lerobot:
            await self._initialize_lerobot()

        # Register all tools (including prediction tools if LeRobot enabled)
        register_tools(
            self.mcp_server, 
            self.ros_bridge, 
            prediction_tools_manager=self._lerobot_tools_manager
        )
        
        # Register all resources (including LeRobot resources if enabled)
        register_resources(
            self.mcp_server, 
            self.ros_bridge,
            lerobot_resource_manager=self._lerobot_resource_manager
        )

        logger.info("MCP server initialized with ROS 2 bridge")

    async def _initialize_lerobot(self) -> None:
        """Initialize LeRobot dataset integration."""
        try:
            from pathlib import Path
            import torch
            from gnn_reasoner import DataManager, LeRobotGraphTransformer, ALOHA_KINEMATIC_CHAIN
            from gnn_reasoner.model import RelationalGNN
            from mcp_ros2_bridge.resources.lerobot_state import LeRobotResourceManager
            from mcp_ros2_bridge.tools.prediction import PredictionToolsManager

            logger.info("Initializing LeRobot integration", repo_id=self.lerobot_repo_id)

            # Initialize components
            data_manager = DataManager(self.lerobot_repo_id, streaming=True)
            graph_transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
            gnn_model = RelationalGNN()

            # Load trained model weights if available
            model_loaded = False
            if self.lerobot_model_path:
                model_path = Path(self.lerobot_model_path)
                if model_path.exists():
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                    gnn_model.load_state_dict(checkpoint["model_state_dict"])
                    gnn_model = gnn_model.to(device)
                    gnn_model.eval()
                    logger.info(
                        "Loaded trained model",
                        path=str(model_path),
                        epoch=checkpoint.get("epoch", "unknown"),
                        val_loss=checkpoint.get("val_loss", "unknown"),
                    )
                    model_loaded = True
                else:
                    logger.warning("Model path not found", path=str(model_path))
            
            # Try auto-detect if no model specified
            if not model_loaded:
                auto_paths = [
                    Path("experiments/aloha_training/best_model.pt"),
                    Path("experiments/training/best_model.pt"),
                ]
                for auto_path in auto_paths:
                    if auto_path.exists():
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        checkpoint = torch.load(auto_path, map_location=device, weights_only=False)
                        gnn_model.load_state_dict(checkpoint["model_state_dict"])
                        gnn_model = gnn_model.to(device)
                        gnn_model.eval()
                        logger.info("Auto-loaded trained model", path=str(auto_path))
                        model_loaded = True
                        break
            
            if not model_loaded:
                logger.info("Using untrained GNN model (random weights)")

            # Create managers (registration happens in initialize())
            self._lerobot_resource_manager = LeRobotResourceManager(
                data_manager, graph_transformer, gnn_model
            )
            self._lerobot_tools_manager = PredictionToolsManager(
                data_manager, graph_transformer, gnn_model
            )

            logger.info("LeRobot components initialized successfully")

        except ImportError as e:
            logger.warning("LeRobot dependencies not available", error=str(e))
        except Exception as e:
            logger.error("Failed to initialize LeRobot integration", error=str(e))

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down MCP-ROS2 Bridge")
        if self.ros_bridge:
            await self.ros_bridge.shutdown()

    def create_app(self) -> Starlette:
        """Create Starlette ASGI application with SSE transport."""
        self.sse_transport = SseServerTransport("/messages/")

        async def handle_sse(request: Request) -> Response:
            """Handle SSE connection from MCP client."""
            logger.info("New MCP client connected", client=request.client)
            async with self.sse_transport.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await self.mcp_server.run(
                    streams[0], streams[1], self.mcp_server.create_initialization_options()
                )
            return Response()

        async def health_check(request: Request) -> Response:
            """Health check endpoint."""
            ros_status = "connected" if self.ros_bridge and self.ros_bridge.is_connected else "disconnected"
            return Response(
                content=f'{{"status": "ok", "ros2": "{ros_status}"}}',
                media_type="application/json"
            )

        @asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncIterator[None]:
            await self.initialize()
            yield
            await self.shutdown()

        # IMPORTANT: Use Mount for handle_post_message, not Route.
        # handle_post_message is an ASGI app, not a Starlette endpoint.
        # Using Route caused "TypeError: 'NoneType' object is not callable"
        # because Route expects the endpoint to return a Response.
        return Starlette(
            debug=True,
            lifespan=lifespan,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages", app=self.sse_transport.handle_post_message),
                Route("/health", endpoint=health_check),
            ],
        )


def main() -> None:
    """Entry point for MCP-ROS2 bridge server."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="MCP-ROS2 Bridge Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument(
        "--lerobot",
        action="store_true",
        help="Enable LeRobot dataset integration",
    )
    parser.add_argument(
        "--lerobot-repo",
        default=None,
        help="LeRobot dataset repo ID (e.g., lerobot/aloha_static_coffee)",
    )
    parser.add_argument(
        "--lerobot-model",
        default=None,
        help="Path to trained GNN model checkpoint (.pt file)",
    )
    args = parser.parse_args()

    server = MCPRos2Server(
        host=args.host,
        port=args.port,
        enable_lerobot=args.lerobot,
        lerobot_repo_id=args.lerobot_repo,
        lerobot_model_path=args.lerobot_model,
    )
    app = server.create_app()

    # Handle graceful shutdown
    def signal_handler(sig: int, frame: object) -> None:
        logger.info("Received shutdown signal", signal=sig)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    uvicorn.run(app, host=server.host, port=server.port, log_level="info")


if __name__ == "__main__":
    main()

