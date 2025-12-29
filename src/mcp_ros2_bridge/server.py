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
from starlette.routing import Route
from starlette.requests import Request
from starlette.responses import Response

from mcp_ros2_bridge.ros_node import ROS2Bridge
from mcp_ros2_bridge.tools import register_tools, register_prediction_tools
from mcp_ros2_bridge.resources import register_resources, register_lerobot_resources

logger = structlog.get_logger()

# LeRobot integration flag - set via environment or constructor
LEROBOT_ENABLED = os.environ.get("LEROBOT_ENABLED", "false").lower() == "true"
LEROBOT_REPO_ID = os.environ.get("LEROBOT_REPO_ID", "lerobot/aloha_static_coffee")


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
    ):
        self.host = host
        self.port = port
        self.enable_lerobot = enable_lerobot or LEROBOT_ENABLED
        self.lerobot_repo_id = lerobot_repo_id or LEROBOT_REPO_ID

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

        # Register core ROS 2 tools and resources
        register_tools(self.mcp_server, self.ros_bridge)
        register_resources(self.mcp_server, self.ros_bridge)

        # Optionally initialize LeRobot integration
        if self.enable_lerobot:
            await self._initialize_lerobot()

        logger.info("MCP server initialized with ROS 2 bridge")

    async def _initialize_lerobot(self) -> None:
        """Initialize LeRobot dataset integration."""
        try:
            from gnn_reasoner import DataManager, LeRobotGraphTransformer, ALOHA_KINEMATIC_CHAIN
            from gnn_reasoner.model import RelationalGNN
            from mcp_ros2_bridge.resources.lerobot_state import LeRobotResourceManager
            from mcp_ros2_bridge.tools.prediction import PredictionToolsManager

            logger.info("Initializing LeRobot integration", repo_id=self.lerobot_repo_id)

            # Initialize components
            data_manager = DataManager(self.lerobot_repo_id, streaming=True)
            graph_transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
            gnn_model = RelationalGNN()

            # Create managers
            self._lerobot_resource_manager = LeRobotResourceManager(
                data_manager, graph_transformer, gnn_model
            )
            self._lerobot_tools_manager = PredictionToolsManager(
                data_manager, graph_transformer, gnn_model
            )

            # Register with MCP server
            register_lerobot_resources(self.mcp_server, self._lerobot_resource_manager)
            register_prediction_tools(self.mcp_server, self._lerobot_tools_manager)

            logger.info("LeRobot integration initialized successfully")

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

        async def handle_messages(request: Request) -> Response:
            """Handle POST messages from MCP client."""
            return await self.sse_transport.handle_post_message(
                request.scope, request.receive, request._send
            )

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

        return Starlette(
            debug=True,
            lifespan=lifespan,
            routes=[
                Route("/sse", endpoint=handle_sse),
                Route("/messages/", endpoint=handle_messages, methods=["POST"]),
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
    args = parser.parse_args()

    server = MCPRos2Server(
        host=args.host,
        port=args.port,
        enable_lerobot=args.lerobot,
        lerobot_repo_id=args.lerobot_repo,
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

