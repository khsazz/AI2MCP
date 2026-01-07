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


SYSTEM_PROMPT = """You are an AI agent controlling a robot through the Model Context Protocol (MCP).

## Your Capabilities

### Motion Tools
- `move(linear_x, angular_z, duration_ms)` - Move with velocity for duration
- `stop()` - Emergency stop
- `rotate(angle_degrees)` - Rotate in place
- `move_forward(distance_meters)` - Move forward by distance

### Perception Tools
- `get_obstacle_distances(directions)` - Query obstacle distances
- `check_path_clear(distance, width)` - Check if path is navigable
- `scan_surroundings(num_sectors)` - 360° obstacle scan

### Prediction Tools (GNN-based reasoning)
- `get_world_graph(threshold)` - Get semantic graph with spatial/interaction predicates
- `get_predicates(threshold)` - Get active predicates (is_near, is_holding, etc.)
- `advance_frame()` - Move to next frame in trajectory
- `set_frame(frame_idx)` - Jump to specific frame
- `predict_action_outcome(action, num_steps)` - Predict future state changes

### Resources (via observations)
- `robot://pose` - Current position (x, y, θ)
- `robot://lerobot/world_graph` - GNN-processed relational graph with predicates
- `robot://lerobot/predicates` - Active predicates with confidence scores

## Reasoning Guidelines

1. **Think step-by-step**: Before acting, analyze the current state and predicates
2. **Use predicates**: The GNN provides semantic understanding:
   - Spatial: `is_near`, `is_above`, `is_below`, `is_left_of`, `is_right_of`
   - Interaction: `is_holding`, `is_contacting`, `is_approaching`, `is_retracting`
3. **Check before moving**: Query obstacles or predicates before motion
4. **Small incremental actions**: Prefer small movements and frequent observation
5. **Stop on risk**: Immediately stop if collision risk detected

## Response Format

Respond with valid JSON only:
- Task complete: `{"action": "complete", "reason": "explanation"}`
- Tool call: `{"action": "tool_call", "tool": "tool_name", "arguments": {...}, "reasoning": "brief explanation"}`

Always include a "reasoning" field explaining your decision.
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
            f"## Step {self.state.step_count} of {self.config.max_steps}\n",
        ]
        
        # Include conversation history for context
        if self.state.message_history:
            prompt_parts.append("## Recent History")
            for turn in self.state.message_history[-self.state.max_history_turns:]:
                prompt_parts.append(f"\n### Step {turn.get('step', '?')}")
                if turn.get('action'):
                    prompt_parts.append(f"Action: `{turn['action'].get('tool', 'unknown')}({turn['action'].get('arguments', {})})`")
                if turn.get('result'):
                    result_summary = str(turn['result'])[:200]
                    prompt_parts.append(f"Result: {result_summary}...")
            prompt_parts.append("")
        
        # Current observation
        prompt_parts.append("## Current Observation\n```json\n" + json.dumps(observation, indent=2) + "\n```\n")

        # Available tools (grouped by category)
        prompt_parts.append("## Available Tools")
        motion_tools = [t for t in available_tools if t['name'] in ('move', 'stop', 'rotate', 'move_forward', 'set_velocity')]
        perception_tools = [t for t in available_tools if t['name'] in ('get_obstacle_distances', 'check_path_clear', 'scan_surroundings')]
        prediction_tools = [t for t in available_tools if t['name'] in ('get_world_graph', 'get_predicates', 'advance_frame', 'set_frame', 'predict_action_outcome')]
        
        if motion_tools:
            prompt_parts.append("**Motion:**")
            prompt_parts.extend(f"- `{t['name']}`: {t.get('description', '')}" for t in motion_tools)
        if perception_tools:
            prompt_parts.append("**Perception:**")
            prompt_parts.extend(f"- `{t['name']}`: {t.get('description', '')}" for t in perception_tools)
        if prediction_tools:
            prompt_parts.append("**Prediction (GNN):**")
            prompt_parts.extend(f"- `{t['name']}`: {t.get('description', '')}" for t in prediction_tools)

        prompt_parts.append(
            "\n## Your Response\n"
            "Analyze the situation step-by-step, then respond with JSON:\n"
            '- `{"action": "complete", "reason": "..."}` if goal achieved\n'
            '- `{"action": "tool_call", "tool": "name", "arguments": {...}, "reasoning": "..."}` for next action'
        )

        return "\n".join(prompt_parts)
    
    def _add_to_history(self, observation: dict, action: dict | None, result: Any) -> None:
        """Add a turn to conversation history."""
        self.state.message_history.append({
            "step": self.state.step_count,
            "observation_summary": {
                "pose": observation.get("pose", {}),
                "predicates_count": len(observation.get("predicates", [])),
            },
            "action": action,
            "result": result,
        })
        
        # Trim to max history
        if len(self.state.message_history) > self.state.max_history_turns * 2:
            self.state.message_history = self.state.message_history[-self.state.max_history_turns:]

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

