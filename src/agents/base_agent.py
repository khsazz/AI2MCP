"""Base agent interface for MCP-based robot control.

Defines the abstract interface that all AI agents must implement,
ensuring consistent behavior regardless of the underlying LLM.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
import structlog
from httpx_sse import aconnect_sse

logger = structlog.get_logger()


@dataclass
class AgentConfig:
    """Configuration for an MCP agent."""
    
    mcp_server_url: str = "http://localhost:8080"
    name: str = "base_agent"
    max_steps: int = 50
    timeout_seconds: float = 30.0
    verbose: bool = True


@dataclass
class AgentState:
    """Current state of the agent."""
    
    step_count: int = 0
    last_observation: dict = field(default_factory=dict)
    last_action: dict = field(default_factory=dict)
    goal: str = ""
    is_complete: bool = False
    error: str | None = None
    
    # Conversation history for context
    message_history: list = field(default_factory=list)
    max_history_turns: int = 5  # Keep last N observation-action pairs


@dataclass
class ToolCall:
    """Represents a tool call to be executed."""
    
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result from executing a tool."""
    
    success: bool
    content: Any
    error: str | None = None


class MCPClient:
    """Client for communicating with MCP server over SSE.
    
    Uses the official MCP SDK for proper protocol communication.
    """

    def __init__(self, server_url: str, timeout: float = 30.0):
        """Initialize MCP client.
        
        Args:
            server_url: Base URL of MCP server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.http_client = httpx.AsyncClient(timeout=timeout)
        self._session: Any = None
        self._read_stream: Any = None
        self._write_stream: Any = None
        self._context_manager: Any = None
        self._session_context: Any = None
        self._tools_cache: list[dict] = []
        self._resources_cache: list[dict] = []

    async def connect(self) -> bool:
        """Establish connection to MCP server using SSE."""
        try:
            # First check health endpoint
            response = await self.http_client.get(f"{self.server_url}/health")
            if response.status_code != 200:
                logger.error("MCP server health check failed", status=response.status_code)
                return False
            
            # Connect via MCP SDK
            try:
                from mcp import ClientSession
                from mcp.client.sse import sse_client
                
                sse_url = f"{self.server_url}/sse"
                self._context_manager = sse_client(sse_url)
                streams = await self._context_manager.__aenter__()
                self._read_stream, self._write_stream = streams
                
                self._session = ClientSession(self._read_stream, self._write_stream)
                self._session_context = await self._session.__aenter__()
                await self._session.initialize()
                
                logger.info("Connected to MCP server via SSE", url=self.server_url)
                
                # Pre-cache tools and resources
                await self._cache_tools_and_resources()
                
                return True
                
            except ImportError:
                logger.warning("MCP SDK not available, using HTTP fallback")
                logger.info("Connected to MCP server (HTTP mode)", url=self.server_url)
                return True
                
        except Exception as e:
            logger.error("Failed to connect to MCP server", error=str(e))
            return False

    async def _cache_tools_and_resources(self) -> None:
        """Cache tools and resources from the server."""
        if self._session is None:
            return
            
        try:
            # Get tools
            tools_result = await self._session.list_tools()
            self._tools_cache = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "inputSchema": t.inputSchema if hasattr(t, 'inputSchema') else {"type": "object", "properties": {}},
                }
                for t in tools_result.tools
            ]
            logger.info("Cached MCP tools", count=len(self._tools_cache))
            
            # Get resources
            resources_result = await self._session.list_resources()
            self._resources_cache = [
                {
                    "uri": str(r.uri),
                    "name": r.name if hasattr(r, 'name') else str(r.uri).split("/")[-1],
                    "description": r.description if hasattr(r, 'description') else "",
                }
                for r in resources_result.resources
            ]
            logger.info("Cached MCP resources", count=len(self._resources_cache))
            
        except Exception as e:
            logger.warning("Failed to cache tools/resources", error=str(e))

    async def list_tools(self) -> list[dict]:
        """List available tools from MCP server."""
        if self._tools_cache:
            return self._tools_cache
            
        # Fallback for HTTP-only mode
        return [
            {"name": "move", "description": "Move robot with velocity", "inputSchema": {"type": "object", "properties": {"linear_x": {"type": "number"}, "angular_z": {"type": "number"}, "duration_ms": {"type": "integer"}}}},
            {"name": "stop", "description": "Emergency stop", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "rotate", "description": "Rotate by angle", "inputSchema": {"type": "object", "properties": {"angle_degrees": {"type": "number"}}}},
            {"name": "move_forward", "description": "Move forward by distance", "inputSchema": {"type": "object", "properties": {"distance_meters": {"type": "number"}}}},
            {"name": "get_obstacle_distances", "description": "Get distances to obstacles", "inputSchema": {"type": "object", "properties": {"directions": {"type": "array", "items": {"type": "string"}}}}},
            {"name": "check_path_clear", "description": "Check if path is clear", "inputSchema": {"type": "object", "properties": {"distance_meters": {"type": "number"}, "width_meters": {"type": "number"}}}},
            {"name": "scan_surroundings", "description": "360-degree obstacle scan", "inputSchema": {"type": "object", "properties": {"num_sectors": {"type": "integer"}}}},
            {"name": "get_world_graph", "description": "Get GNN world graph with predicates", "inputSchema": {"type": "object", "properties": {"threshold": {"type": "number", "default": 0.5}}}},
            {"name": "get_predicates", "description": "Get active spatial/interaction predicates", "inputSchema": {"type": "object", "properties": {"threshold": {"type": "number", "default": 0.5}}}},
            {"name": "advance_frame", "description": "Advance to next frame in trajectory", "inputSchema": {"type": "object", "properties": {}}},
            {"name": "set_frame", "description": "Set current frame index", "inputSchema": {"type": "object", "properties": {"frame_idx": {"type": "integer"}}}},
        ]

    async def list_resources(self) -> list[dict]:
        """List available resources from MCP server."""
        if self._resources_cache:
            return self._resources_cache
            
        # Fallback for HTTP-only mode
        return [
            {"uri": "robot://pose", "description": "Robot position and orientation"},
            {"uri": "robot://velocity", "description": "Robot velocity"},
            {"uri": "robot://scan/summary", "description": "LiDAR scan summary"},
            {"uri": "robot://world_graph", "description": "Semantic world graph"},
            {"uri": "robot://lerobot/current_state", "description": "Current LeRobot observation state"},
            {"uri": "robot://lerobot/world_graph", "description": "GNN-processed world graph with predicates"},
            {"uri": "robot://lerobot/predicates", "description": "Active spatial and interaction predicates"},
        ]

    async def read_resource(self, uri: str) -> dict:
        """Read a resource from MCP server."""
        import json
        try:
            if self._session is not None:
                # Use MCP SDK
                result = await self._session.read_resource(uri)
                
                # Handle different result formats
                contents = None
                if hasattr(result, 'contents'):
                    contents = result.contents
                elif hasattr(result, 'content'):
                    contents = [result.content] if not isinstance(result.content, list) else result.content
                elif isinstance(result, list):
                    contents = result
                else:
                    return {"error": f"Unknown result type: {type(result).__name__}"}
                
                if not contents:
                    return {"error": "Empty resource response"}
                
                for content in contents:
                    text = None
                    
                    # Try multiple ways to get the text content
                    try:
                        # TextContent has .text attribute
                        if hasattr(content, 'text') and content.text is not None:
                            text = content.text
                        # BlobContent has .data
                        elif hasattr(content, 'data') and content.data is not None:
                            text = content.data if isinstance(content.data, str) else str(content.data)
                        # Direct string
                        elif isinstance(content, str):
                            text = content
                        # Last resort: repr
                        elif content is not None:
                            # Check for common JSON-like dict
                            if hasattr(content, '__dict__'):
                                text = json.dumps({k: v for k, v in content.__dict__.items() if not k.startswith('_')})
                    except Exception as inner_e:
                        logger.debug(f"Error extracting content: {inner_e}")
                        continue
                    
                    if text:
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            return {"raw": text[:500]}
                            
                return {"error": "No parseable content"}
            else:
                # HTTP fallback
                response = await self.http_client.get(
                    f"{self.server_url}/resource",
                    params={"uri": uri},
                )
                if response.status_code == 200:
                    return response.json()
                return {"error": f"Failed to read resource: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def _coerce_argument_types(self, arguments: dict) -> dict:
        """Convert string numbers to actual numbers for MCP tools."""
        result = {}
        for key, value in arguments.items():
            if isinstance(value, str):
                # Try to convert to number
                try:
                    if '.' in value:
                        result[key] = float(value)
                    else:
                        result[key] = int(value)
                except ValueError:
                    result[key] = value
            else:
                result[key] = value
        return result

    async def call_tool(self, name: str, arguments: dict) -> ToolResult:
        """Call a tool on the MCP server."""
        # Coerce argument types (LLMs often send "0" instead of 0)
        arguments = self._coerce_argument_types(arguments)
        
        try:
            if self._session is not None:
                # Use MCP SDK
                result = await self._session.call_tool(name, arguments)
                
                # Extract content from result
                content_data = {}
                for content in result.content:
                    text = getattr(content, 'text', None)
                    if text:
                        try:
                            import json
                            content_data = json.loads(text)
                        except json.JSONDecodeError:
                            content_data = {"text": text}
                        break
                
                return ToolResult(
                    success=True,
                    content=content_data,
                )
            else:
                # HTTP fallback
                response = await self.http_client.post(
                    f"{self.server_url}/messages/",
                    json={
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": name,
                            "arguments": arguments,
                        },
                        "id": 1,
                    },
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return ToolResult(
                        success=True,
                        content=result.get("result", {}),
                    )
                return ToolResult(
                    success=False,
                    content=None,
                    error=f"HTTP {response.status_code}",
                )
        except Exception as e:
            return ToolResult(
                success=False,
                content=None,
                error=str(e),
            )

    async def close(self) -> None:
        """Close the client connection."""
        try:
            if self._session is not None:
                await self._session.__aexit__(None, None, None)
            if self._context_manager is not None:
                await self._context_manager.__aexit__(None, None, None)
        except Exception as e:
            logger.warning("Error closing MCP session", error=str(e))
        
        await self.http_client.aclose()


class BaseAgent(ABC):
    """Abstract base class for MCP-based robot control agents."""

    def __init__(self, config: AgentConfig | None = None):
        """Initialize agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config or AgentConfig()
        self.state = AgentState()
        self.mcp_client = MCPClient(
            self.config.mcp_server_url,
            self.config.timeout_seconds,
        )
        self.action_history: list[dict] = []

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the underlying model."""
        ...

    @abstractmethod
    async def decide_action(
        self,
        observation: dict,
        available_tools: list[dict],
    ) -> ToolCall | None:
        """Decide what action to take based on observation.
        
        Args:
            observation: Current state observation
            available_tools: List of available MCP tools
            
        Returns:
            ToolCall to execute, or None if task is complete
        """
        ...

    async def observe(self) -> dict:
        """Gather current observation from robot.
        
        Note: MCP resources have a library bug, so we use tool calls instead.
        The last_action result contains GNN data from tools like get_world_graph.
        """
        observation = {}

        # Use last action result if available (tools work, resources don't)
        if self.state.last_action and self.state.last_action.get("result"):
            result = self.state.last_action["result"]
            if isinstance(result, dict):
                # Extract world context from tool result
                if "world_context" in result:
                    ctx = result["world_context"]
                    observation["world_graph"] = {
                        "num_nodes": ctx.get("num_nodes", 0),
                        "num_edges": ctx.get("num_edges", 0),
                    }
                    # Extract predicates
                    spatial = result.get("spatial_predicates", [])
                    interaction = result.get("interaction_predicates", [])
                    observation["predicates"] = {
                        "spatial_count": len(spatial),
                        "interaction_count": len(interaction),
                    }

        self.state.last_observation = observation
        return observation

    async def execute_action(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call on the robot."""
        logger.info(
            "Executing action",
            tool=tool_call.name,
            arguments=tool_call.arguments,
        )

        result = await self.mcp_client.call_tool(
            tool_call.name,
            tool_call.arguments,
        )

        self.action_history.append({
            "step": self.state.step_count,
            "tool": tool_call.name,
            "arguments": tool_call.arguments,
            "success": result.success,
        })

        self.state.last_action = {
            "tool": tool_call.name,
            "arguments": tool_call.arguments,
            "result": result.content,
        }

        return result

    async def run(self, goal: str) -> AsyncIterator[dict]:
        """Run agent until goal is achieved or max steps reached.
        
        Args:
            goal: Natural language description of the goal
            
        Yields:
            Step-by-step progress updates
        """
        self.state.goal = goal
        self.state.step_count = 0
        self.state.is_complete = False

        # Connect to MCP server
        if not await self.mcp_client.connect():
            self.state.error = "Failed to connect to MCP server"
            yield {"status": "error", "error": self.state.error}
            return

        # Get available tools
        tools = await self.mcp_client.list_tools()

        logger.info(
            "Starting agent run",
            model=self.model_name,
            goal=goal,
            available_tools=len(tools),
        )

        yield {
            "status": "started",
            "model": self.model_name,
            "goal": goal,
            "available_tools": [t["name"] for t in tools],
        }

        while self.state.step_count < self.config.max_steps and not self.state.is_complete:
            self.state.step_count += 1

            try:
                # Observe
                observation = await self.observe()

                yield {
                    "status": "observing",
                    "step": self.state.step_count,
                    "observation": observation,
                }

                # Decide
                action = await self.decide_action(observation, tools)

                if action is None:
                    self.state.is_complete = True
                    yield {
                        "status": "complete",
                        "step": self.state.step_count,
                        "reason": "Agent decided task is complete",
                    }
                    break

                yield {
                    "status": "deciding",
                    "step": self.state.step_count,
                    "action": {
                        "tool": action.name,
                        "arguments": action.arguments,
                    },
                }

                # Execute
                result = await self.execute_action(action)

                yield {
                    "status": "executed",
                    "step": self.state.step_count,
                    "result": {
                        "success": result.success,
                        "content": result.content,
                        "error": result.error,
                    },
                }

            except Exception as e:
                logger.error("Agent step failed", error=str(e))
                self.state.error = str(e)
                yield {
                    "status": "error",
                    "step": self.state.step_count,
                    "error": str(e),
                }
                break

        if self.state.step_count >= self.config.max_steps and not self.state.is_complete:
            yield {
                "status": "max_steps_reached",
                "step": self.state.step_count,
            }

        # Cleanup
        await self.mcp_client.close()

    async def run_single_action(self, tool_name: str, arguments: dict) -> ToolResult:
        """Execute a single action without full agent loop.
        
        Useful for testing or manual control.
        """
        if not await self.mcp_client.connect():
            return ToolResult(
                success=False,
                content=None,
                error="Failed to connect to MCP server",
            )

        result = await self.execute_action(ToolCall(name=tool_name, arguments=arguments))
        await self.mcp_client.close()
        return result

