"""MCP Server with SSE transport for ROS 2 integration.

This module implements the main MCP server that exposes robot capabilities
as Tools (actions) and Resources (state) over Server-Sent Events.
"""

from __future__ import annotations

import asyncio
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
from mcp_ros2_bridge.tools import register_tools
from mcp_ros2_bridge.resources import register_resources

logger = structlog.get_logger()


class MCPRos2Server:
    """MCP Server bridging AI models to ROS 2."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.mcp_server = Server("mcp-ros2-bridge")
        self.ros_bridge: ROS2Bridge | None = None
        self.sse_transport: SseServerTransport | None = None

    async def initialize(self) -> None:
        """Initialize ROS 2 bridge and register MCP handlers."""
        logger.info("Initializing MCP-ROS2 Bridge", host=self.host, port=self.port)
        
        # Initialize ROS 2 bridge
        self.ros_bridge = ROS2Bridge()
        await self.ros_bridge.initialize()
        
        # Register tools and resources
        register_tools(self.mcp_server, self.ros_bridge)
        register_resources(self.mcp_server, self.ros_bridge)
        
        logger.info("MCP server initialized with ROS 2 bridge")

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
    import uvicorn

    server = MCPRos2Server(
        host="0.0.0.0",
        port=8080,
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

