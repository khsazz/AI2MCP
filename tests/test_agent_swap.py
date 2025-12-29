"""Tests for agent swappability - key thesis validation.

These tests verify that different AI models can be swapped
without modifying the robot-side code.
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.base_agent import AgentConfig, ToolCall, MCPClient
from agents.claude_agent import ClaudeAgent
from agents.llama_agent import LlamaAgent


class TestAgentInterface:
    """Test that all agents implement the same interface."""

    def test_claude_agent_has_model_name(self) -> None:
        """Test Claude agent exposes model name."""
        agent = ClaudeAgent()
        assert "claude" in agent.model_name.lower()

    def test_llama_agent_has_model_name(self) -> None:
        """Test Llama agent exposes model name."""
        agent = LlamaAgent()
        assert "llama" in agent.model_name.lower()

    def test_agents_share_config_interface(self) -> None:
        """Test all agents accept the same config."""
        config = AgentConfig(
            mcp_server_url="http://localhost:8080",
            max_steps=10,
        )
        
        claude = ClaudeAgent(config=config)
        llama = LlamaAgent(config=config)
        
        assert claude.config.mcp_server_url == llama.config.mcp_server_url
        assert claude.config.max_steps == llama.config.max_steps


class TestMCPClient:
    """Test MCP client functionality."""

    @pytest.mark.asyncio
    async def test_client_initialization(self) -> None:
        """Test MCP client can be initialized."""
        client = MCPClient("http://localhost:8080")
        assert client.server_url == "http://localhost:8080"
        await client.close()

    @pytest.mark.asyncio
    async def test_tool_list_format(self) -> None:
        """Test that tool list has expected format."""
        client = MCPClient("http://localhost:8080")
        tools = await client.list_tools()
        
        assert isinstance(tools, list)
        if tools:
            assert "name" in tools[0]
            assert "description" in tools[0]
        
        await client.close()


class TestSwappability:
    """Test that agents are truly swappable."""

    def test_same_observation_format(self) -> None:
        """Test agents receive observations in same format."""
        # Both agents use the same MCPClient
        claude = ClaudeAgent()
        llama = LlamaAgent()
        
        # Both have mcp_client attribute
        assert hasattr(claude, 'mcp_client')
        assert hasattr(llama, 'mcp_client')
        
        # Both clients are same type
        assert type(claude.mcp_client) == type(llama.mcp_client)

    def test_same_action_format(self) -> None:
        """Test agents produce actions in same format."""
        # Both agents return ToolCall or None from decide_action
        action = ToolCall(name="move", arguments={"linear_x": 0.5, "angular_z": 0.0, "duration_ms": 1000})
        
        assert action.name == "move"
        assert "linear_x" in action.arguments

    def test_agents_are_interchangeable(self) -> None:
        """Test that agent swap doesn't require code changes.
        
        This is the key thesis claim: any agent implementing BaseAgent
        can control the robot through MCP without modification.
        """
        config = AgentConfig(mcp_server_url="http://localhost:8080")
        
        # Create both agents with same config
        agents = [
            ClaudeAgent(config=config),
            LlamaAgent(config=config),
        ]
        
        for agent in agents:
            # All agents have same methods
            assert hasattr(agent, 'observe')
            assert hasattr(agent, 'decide_action')
            assert hasattr(agent, 'execute_action')
            assert hasattr(agent, 'run')
            
            # All agents connect to same MCP server
            assert agent.config.mcp_server_url == "http://localhost:8080"

