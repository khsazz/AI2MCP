"""Qwen-based agent for MCP robot control.

Demonstrates using Alibaba's Qwen models (via Ollama)
as a local "brain" for robot control through the MCP interface.

Qwen2.5 models are excellent at instruction following and JSON output.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import structlog

from agents.base_agent import BaseAgent, AgentConfig, ToolCall

logger = structlog.get_logger()


# Qwen-optimized system prompt
QWEN_SYSTEM_PROMPT = """You control a robot via MCP (Model Context Protocol).

RESPOND WITH EXACTLY ONE JSON:

If predicates found (spatial > 0):
{"action": "complete", "reason": "Found N spatial predicates"}

If no predicates yet:
{"action": "tool_call", "tool": "get_world_graph", "arguments": {"threshold": 0.5}}

IMPORTANT: Once you see "Spatial predicates detected: N" where N > 0, you MUST return complete.

JSON ONLY. No markdown."""


class QwenAgent(BaseAgent):
    """Agent using Qwen models via Ollama for decision making.
    
    Qwen2.5 models are particularly good at:
    - Instruction following
    - JSON output formatting
    - Tool use reasoning
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        model: str = "qwen2.5:3b",
        ollama_url: str = "http://localhost:11434",
    ):
        """Initialize Qwen agent.
        
        Args:
            config: Agent configuration
            model: Ollama model name (e.g., "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b")
            ollama_url: Ollama API endpoint
        """
        super().__init__(config)
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=60.0)

    @property
    def model_name(self) -> str:
        return f"qwen/{self.model}"

    async def _check_ollama(self) -> bool:
        """Check if Ollama is available and model is loaded."""
        try:
            response = await self._client.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                return False
            
            # Check if our model is available
            tags = response.json()
            models = [m["name"] for m in tags.get("models", [])]
            if self.model not in models and f"{self.model}:latest" not in models:
                logger.warning(f"Model {self.model} not found in Ollama", available=models)
            return True
        except Exception as e:
            logger.error("Ollama check failed", error=str(e))
            return False

    def _build_prompt(
        self,
        observation: dict,
        available_tools: list[dict],
    ) -> str:
        """Build prompt for Qwen with current context."""
        # Format tools list
        tools_desc = "\n".join(
            f"- {t['name']}: {t.get('description', 'No description')}"
            for t in available_tools[:8]  # Limit to avoid context overflow
        )
        
        # Build history section
        history_section = ""
        if self.state.message_history:
            history_lines = []
            for turn in self.state.message_history[-3:]:
                action = turn.get('action', {})
                if action:
                    history_lines.append(
                        f"Step {turn.get('step')}: Called {action.get('tool', '?')} "
                        f"-> {'success' if turn.get('result_success') else 'failed'}"
                    )
            if history_lines:
                history_section = "Previous actions:\n" + "\n".join(history_lines) + "\n\n"

        # Extract key observation data
        predicates = observation.get("predicates", {})
        spatial_count = predicates.get("spatial_count", 0)
        interaction_count = predicates.get("interaction_count", 0)
        world_graph = observation.get("world_graph", {})
        
        # Last action result summary
        last_result = ""
        if self.state.last_action:
            tool_name = self.state.last_action.get("tool", "unknown")
            result = self.state.last_action.get("result", {})
            if isinstance(result, dict):
                if "predicates" in result:
                    preds = result["predicates"]
                    last_result = f"Last action: {tool_name} returned {len(preds)} predicates"
                elif "world_context" in result:
                    ctx = result["world_context"]
                    last_result = f"Last action: {tool_name} returned graph with {ctx.get('num_nodes', '?')} nodes"
                else:
                    last_result = f"Last action: {tool_name} completed successfully"

        # Log observation data for debugging
        logger.debug(
            "Building Qwen prompt",
            step=self.state.step_count,
            spatial_count=spatial_count,
            interaction_count=interaction_count,
            num_nodes=world_graph.get('num_nodes', 'unknown'),
        )

        # Build the prompt - be very explicit about when to complete
        if spatial_count > 0:
            prompt = f"""Goal: {self.state.goal}

RESULT: Found {spatial_count} spatial predicates and {interaction_count} interaction predicates.
Graph has {world_graph.get('num_nodes', 'unknown')} nodes.

Task is COMPLETE. Return:
{{"action": "complete", "reason": "Found {spatial_count} spatial predicates"}}"""
        else:
            prompt = f"""Goal: {self.state.goal}

Step: {self.state.step_count} / {self.config.max_steps}

No predicates detected yet. Call get_world_graph to analyze the scene.

Return:
{{"action": "tool_call", "tool": "get_world_graph", "arguments": {{"threshold": 0.5}}}}"""

        return prompt

    def _add_to_history(self, observation: dict, action: dict | None, result: Any) -> None:
        """Add a turn to conversation history."""
        self.state.message_history.append({
            "step": self.state.step_count,
            "action": action,
            "result_success": result.success if hasattr(result, 'success') else True,
        })
        
        # Trim history to save context
        if len(self.state.message_history) > 6:
            self.state.message_history = self.state.message_history[-3:]

    async def decide_action(
        self,
        observation: dict,
        available_tools: list[dict],
    ) -> ToolCall | None:
        """Use Qwen to decide the next action."""
        if not await self._check_ollama():
            logger.error("Ollama not available", url=self.ollama_url)
            return None

        prompt = self._build_prompt(observation, available_tools)

        try:
            # Use Ollama's chat API for better Qwen compatibility
            response = await self._client.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": QWEN_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Lower for more deterministic JSON
                        "top_p": 0.9,
                        "num_predict": 256,  # Limit output length
                    },
                },
            )

            if response.status_code != 200:
                logger.error("Ollama request failed", status=response.status_code, body=response.text)
                return None

            result = response.json()
            response_text = result.get("message", {}).get("content", "")
            logger.debug("Qwen response", response=response_text[:200])

            # Parse JSON from response (Qwen sometimes adds markdown)
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                decision = json.loads(json_str)
            else:
                logger.error("No JSON found in Qwen response", response=response_text[:100])
                return None

            # Handle complete action
            if decision.get("action") == "complete":
                logger.info("Qwen decided task is complete", reason=decision.get("reason"))
                return None

            # Handle tool call
            if decision.get("action") == "tool_call":
                tool_name = decision.get("tool")
                arguments = decision.get("arguments", {})
                
                logger.info("Qwen decided to call tool", tool=tool_name, args=arguments)
                return ToolCall(
                    name=tool_name,
                    arguments=arguments,
                )

            logger.warning("Unknown action type from Qwen", decision=decision)
            return None

        except json.JSONDecodeError as e:
            logger.error("Failed to parse Qwen JSON response", error=str(e))
            return None
        except Exception as e:
            logger.error("Qwen API call failed", error=str(e))
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
