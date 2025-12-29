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
    """Client for communicating with MCP server over SSE."""

    def __init__(self, server_url: str, timeout: float = 30.0):
        """Initialize MCP client.
        
        Args:
            server_url: Base URL of MCP server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        self._session_id: str | None = None

    async def connect(self) -> bool:
        """Establish connection to MCP server."""
        try:
            response = await self.client.get(f"{self.server_url}/health")
            if response.status_code == 200:
                logger.info("Connected to MCP server", url=self.server_url)
                return True
            return False
        except Exception as e:
            logger.error("Failed to connect to MCP server", error=str(e))
            return False

    async def list_tools(self) -> list[dict]:
        """List available tools from MCP server."""
        # In full implementation, this would use MCP protocol
        # For now, return mock tool list
        return [
            {"name": "move", "description": "Move robot with velocity"},
            {"name": "stop", "description": "Emergency stop"},
            {"name": "rotate", "description": "Rotate by angle"},
            {"name": "move_forward", "description": "Move forward by distance"},
            {"name": "get_obstacle_distances", "description": "Get distances to obstacles"},
            {"name": "check_path_clear", "description": "Check if path is clear"},
            {"name": "scan_surroundings", "description": "360-degree obstacle scan"},
        ]

    async def list_resources(self) -> list[dict]:
        """List available resources from MCP server."""
        return [
            {"uri": "robot://pose", "description": "Robot position and orientation"},
            {"uri": "robot://velocity", "description": "Robot velocity"},
            {"uri": "robot://scan/summary", "description": "LiDAR scan summary"},
            {"uri": "robot://world_graph", "description": "Semantic world graph"},
        ]

    async def read_resource(self, uri: str) -> dict:
        """Read a resource from MCP server."""
        try:
            # This would use MCP protocol in full implementation
            response = await self.client.get(
                f"{self.server_url}/resource",
                params={"uri": uri},
            )
            if response.status_code == 200:
                return response.json()
            return {"error": f"Failed to read resource: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    async def call_tool(self, name: str, arguments: dict) -> ToolResult:
        """Call a tool on the MCP server."""
        try:
            response = await self.client.post(
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
        await self.client.aclose()


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
        """Gather current observation from robot."""
        observation = {}

        # Read key resources
        resources = ["robot://pose", "robot://scan/summary", "robot://world_graph"]
        for uri in resources:
            data = await self.mcp_client.read_resource(uri)
            key = uri.split("/")[-1]
            observation[key] = data

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

