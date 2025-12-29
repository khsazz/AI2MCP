"""Claude-based agent for MCP robot control.

Demonstrates using Anthropic's Claude models as the "brain"
for robot control through the MCP interface.
"""

from __future__ import annotations

import json
import os
from typing import Any

import structlog

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from agents.base_agent import BaseAgent, AgentConfig, ToolCall

logger = structlog.get_logger()


SYSTEM_PROMPT = """You are an AI agent controlling a mobile robot through the Model Context Protocol (MCP).

Your capabilities:
- Move the robot using velocity commands (move, set_velocity, stop)
- Navigate using high-level commands (move_forward, rotate)
- Query sensor data (get_obstacle_distances, check_path_clear, scan_surroundings)
- Access robot state through resources (pose, velocity, scan data, world graph)

Guidelines:
1. Always check for obstacles before moving
2. Use scan_surroundings to understand the environment
3. Move carefully - prefer small movements and checking often
4. Stop immediately if you detect a potential collision
5. Explain your reasoning briefly before each action

Current observations and world state will be provided. Analyze them to decide your next action.
If the goal is achieved, respond with {"action": "complete", "reason": "..."}.
Otherwise, respond with {"action": "tool_call", "tool": "tool_name", "arguments": {...}}.
"""


class ClaudeAgent(BaseAgent):
    """Agent using Claude for decision making."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ):
        """Initialize Claude agent.
        
        Args:
            config: Agent configuration
            model: Claude model to use
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
        """
        super().__init__(config)
        self.model = model
        self._client: anthropic.Anthropic | None = None

        if ANTHROPIC_AVAILABLE:
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = anthropic.Anthropic(api_key=api_key)
            else:
                logger.warning("No Anthropic API key provided")
        else:
            logger.warning("anthropic package not installed")

    @property
    def model_name(self) -> str:
        return f"claude/{self.model}"

    def _format_tools_for_claude(self, tools: list[dict]) -> list[dict]:
        """Format MCP tools as Claude tool definitions."""
        claude_tools = []
        for tool in tools:
            claude_tools.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema", {"type": "object", "properties": {}}),
            })
        return claude_tools

    def _build_prompt(
        self,
        observation: dict,
        available_tools: list[dict],
    ) -> str:
        """Build prompt for Claude with current context."""
        prompt_parts = [
            f"## Current Goal\n{self.state.goal}\n",
            f"## Step {self.state.step_count}\n",
            "## Current Observation\n```json\n" + json.dumps(observation, indent=2) + "\n```\n",
        ]

        if self.state.last_action:
            prompt_parts.append(
                "## Last Action\n```json\n" + json.dumps(self.state.last_action, indent=2) + "\n```\n"
            )

        prompt_parts.append(
            "## Available Tools\n" + 
            "\n".join(f"- {t['name']}: {t.get('description', '')}" for t in available_tools)
        )

        prompt_parts.append(
            "\n\nBased on the observation and your goal, decide your next action. "
            "Respond with a JSON object containing either:\n"
            '- {"action": "complete", "reason": "..."} if the goal is achieved\n'
            '- {"action": "tool_call", "tool": "tool_name", "arguments": {...}} for the next action'
        )

        return "\n".join(prompt_parts)

    async def decide_action(
        self,
        observation: dict,
        available_tools: list[dict],
    ) -> ToolCall | None:
        """Use Claude to decide the next action."""
        if self._client is None:
            logger.error("Claude client not initialized")
            return None

        prompt = self._build_prompt(observation, available_tools)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            # Extract response text
            response_text = response.content[0].text
            logger.debug("Claude response", response=response_text)

            # Parse JSON response
            # Find JSON in response (may have explanation before/after)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                decision = json.loads(json_str)
            else:
                logger.error("No JSON found in Claude response")
                return None

            if decision.get("action") == "complete":
                logger.info("Claude decided task is complete", reason=decision.get("reason"))
                return None

            if decision.get("action") == "tool_call":
                return ToolCall(
                    name=decision["tool"],
                    arguments=decision.get("arguments", {}),
                )

            logger.warning("Unknown action type", decision=decision)
            return None

        except Exception as e:
            logger.error("Claude API call failed", error=str(e))
            return None


class ClaudeAgentWithTools(ClaudeAgent):
    """Claude agent using native tool use API."""

    async def decide_action(
        self,
        observation: dict,
        available_tools: list[dict],
    ) -> ToolCall | None:
        """Use Claude's native tool use for decision making."""
        if self._client is None:
            logger.error("Claude client not initialized")
            return None

        # Format tools for Claude's tool use API
        claude_tools = self._format_tools_for_claude(available_tools)

        # Build user message
        user_message = (
            f"Goal: {self.state.goal}\n\n"
            f"Current observation:\n{json.dumps(observation, indent=2)}\n\n"
            "Analyze the situation and call the appropriate tool, or say 'TASK COMPLETE' if done."
        )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=claude_tools,
                messages=[
                    {"role": "user", "content": user_message},
                ],
            )

            # Check for tool use in response
            for block in response.content:
                if block.type == "tool_use":
                    return ToolCall(
                        name=block.name,
                        arguments=block.input,
                    )
                elif block.type == "text":
                    if "TASK COMPLETE" in block.text.upper():
                        logger.info("Claude decided task is complete")
                        return None

            # No tool call - check if stop_reason indicates completion
            if response.stop_reason == "end_turn":
                logger.info("Claude ended turn without tool call")
                return None

            return None

        except Exception as e:
            logger.error("Claude API call failed", error=str(e))
            return None

