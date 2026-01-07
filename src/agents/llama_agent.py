"""Llama-based agent for MCP robot control.

Demonstrates using Meta's Llama models (via Ollama or vLLM)
as a local "brain" for robot control through the MCP interface.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import structlog

from agents.base_agent import BaseAgent, AgentConfig, ToolCall

logger = structlog.get_logger()


SYSTEM_PROMPT = """You control a robot via MCP. Output JSON only.

IMPORTANT: After 1-2 tool calls, you MUST complete the task.
Do NOT keep calling the same tools repeatedly.

## Response Format
Complete task: {"action": "complete", "reason": "Found X predicates"}
Call tool: {"action": "tool_call", "tool": "name", "arguments": {}}

## Available Tools
- get_world_graph(threshold) - Get scene graph
- get_predicates(threshold) - Get active predicates  
- move(linear_x, angular_z, duration_ms) - Move robot
- stop() - Emergency stop

## Predicates (detected by GNN)
Spatial: is_near, is_left_of, is_right_of
Interaction: is_holding, is_contacting

JSON only. Complete after gathering info."""


class LlamaAgent(BaseAgent):
    """Agent using Llama models via Ollama for decision making."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        model: str = "llama3.2",
        ollama_url: str = "http://localhost:11434",
    ):
        """Initialize Llama agent.
        
        Args:
            config: Agent configuration
            model: Ollama model name (e.g., "llama3.2", "llama3.1:70b")
            ollama_url: Ollama API endpoint
        """
        super().__init__(config)
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=60.0)

    @property
    def model_name(self) -> str:
        return f"llama/{self.model}"

    async def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = await self._client.get(f"{self.ollama_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    def _build_prompt(
        self,
        observation: dict,
        available_tools: list[dict],
    ) -> str:
        """Build prompt for Llama with current context."""
        tools_desc = "\n".join(
            f"- {t['name']}: {t.get('description', '')}"
            for t in available_tools
        )
        
        # Build history section
        history_section = ""
        if self.state.message_history:
            history_lines = []
            for turn in self.state.message_history[-3:]:  # Last 3 turns for Llama (shorter context)
                action = turn.get('action', {})
                if action:
                    history_lines.append(f"Step {turn.get('step')}: {action.get('tool', '?')}({action.get('arguments', {})})")
            if history_lines:
                history_section = "Recent actions:\n" + "\n".join(history_lines) + "\n\n"

        # Check if we have predicate data (goal achieved indicator)
        predicates = observation.get("predicates", {})
        spatial_count = predicates.get("spatial_count", 0)
        interaction_count = predicates.get("interaction_count", 0)
        
        # Last action result
        last_result = ""
        if self.state.last_action:
            tool_name = self.state.last_action.get("tool", "unknown")
            result = self.state.last_action.get("result", {})
            if result:
                # Summarize result
                if isinstance(result, dict):
                    if "world_context" in result:
                        ctx = result["world_context"]
                        last_result = f"Last result: {tool_name} returned {ctx.get('num_nodes', '?')} nodes, {ctx.get('num_edges', '?')} edges"
                    elif "predicates" in result:
                        last_result = f"Last result: {tool_name} returned {len(result['predicates'])} predicates"
                    else:
                        last_result = f"Last result: {tool_name} succeeded"
                else:
                    last_result = f"Last result: {tool_name} returned data"

        return f"""Goal: {self.state.goal}

Step: {self.state.step_count}/{self.config.max_steps}
{last_result}

{history_section}Current observation:
- Spatial predicates detected: {spatial_count}
- Interaction predicates detected: {interaction_count}
- World graph: {observation.get("world_graph", {}).get("num_nodes", "?")} nodes

DECISION RULE:
- If spatial_count > 0: Task done! Return complete.
- If step > 2: Stop and report what you found.
- Otherwise: Call get_world_graph once.

Example complete response:
{{"action": "complete", "reason": "Found {spatial_count} spatial predicates"}}

JSON only:"""
    
    def _add_to_history(self, observation: dict, action: dict | None, result: Any) -> None:
        """Add a turn to conversation history."""
        self.state.message_history.append({
            "step": self.state.step_count,
            "action": action,
            "result_success": result.success if hasattr(result, 'success') else True,
        })
        
        # Trim to max history
        if len(self.state.message_history) > 6:
            self.state.message_history = self.state.message_history[-3:]

    async def decide_action(
        self,
        observation: dict,
        available_tools: list[dict],
    ) -> ToolCall | None:
        """Use Llama to decide the next action."""
        if not await self._check_ollama():
            logger.error("Ollama not available", url=self.ollama_url)
            return None

        prompt = self._build_prompt(observation, available_tools)

        try:
            response = await self._client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                    },
                },
            )

            if response.status_code != 200:
                logger.error("Ollama request failed", status=response.status_code)
                return None

            result = response.json()
            response_text = result.get("response", "")
            logger.debug("Llama response", response=response_text)

            # Parse JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                decision = json.loads(json_str)
            else:
                logger.error("No JSON found in Llama response")
                return None

            if decision.get("action") == "complete":
                logger.info("Llama decided task is complete", reason=decision.get("reason"))
                return None

            if decision.get("action") == "tool_call":
                return ToolCall(
                    name=decision["tool"],
                    arguments=decision.get("arguments", {}),
                )

            logger.warning("Unknown action type", decision=decision)
            return None

        except json.JSONDecodeError as e:
            logger.error("Failed to parse Llama JSON response", error=str(e))
            return None
        except Exception as e:
            logger.error("Llama API call failed", error=str(e))
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


class VLLMAgent(BaseAgent):
    """Agent using Llama models via vLLM for high-performance inference."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
        vllm_url: str = "http://localhost:8000",
    ):
        """Initialize vLLM agent.
        
        Args:
            config: Agent configuration
            model: Model name/path
            vllm_url: vLLM OpenAI-compatible API endpoint
        """
        super().__init__(config)
        self.model = model
        self.vllm_url = vllm_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=60.0)

    @property
    def model_name(self) -> str:
        return f"vllm/{self.model.split('/')[-1]}"

    def _build_messages(
        self,
        observation: dict,
        available_tools: list[dict],
    ) -> list[dict]:
        """Build chat messages for vLLM."""
        tools_desc = "\n".join(
            f"- {t['name']}: {t.get('description', '')}"
            for t in available_tools
        )

        user_content = f"""Goal: {self.state.goal}

Step: {self.state.step_count}

Available tools:
{tools_desc}

Current observation:
{json.dumps(observation, indent=2)}

{f"Last action: {json.dumps(self.state.last_action)}" if self.state.last_action else ""}

Respond with JSON only."""

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    async def decide_action(
        self,
        observation: dict,
        available_tools: list[dict],
    ) -> ToolCall | None:
        """Use vLLM to decide the next action."""
        messages = self._build_messages(observation, available_tools)

        try:
            response = await self._client.post(
                f"{self.vllm_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 512,
                },
            )

            if response.status_code != 200:
                logger.error("vLLM request failed", status=response.status_code)
                return None

            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            logger.debug("vLLM response", response=response_text)

            # Parse JSON
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                decision = json.loads(response_text[json_start:json_end])
            else:
                logger.error("No JSON found in vLLM response")
                return None

            if decision.get("action") == "complete":
                logger.info("vLLM decided task is complete")
                return None

            if decision.get("action") == "tool_call":
                return ToolCall(
                    name=decision["tool"],
                    arguments=decision.get("arguments", {}),
                )

            return None

        except Exception as e:
            logger.error("vLLM API call failed", error=str(e))
            return None

