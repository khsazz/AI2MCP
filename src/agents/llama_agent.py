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


SYSTEM_PROMPT = """You are an AI agent controlling a mobile robot through the Model Context Protocol (MCP).

Your capabilities:
- Move the robot using velocity commands (move, set_velocity, stop)
- Navigate using high-level commands (move_forward, rotate)
- Query sensor data (get_obstacle_distances, check_path_clear, scan_surroundings)

Guidelines:
1. Always check for obstacles before moving
2. Use scan_surroundings to understand the environment
3. Move carefully with small movements
4. Stop immediately if collision risk detected

You must respond with valid JSON only. Format:
- Task complete: {"action": "complete", "reason": "explanation"}
- Tool call: {"action": "tool_call", "tool": "tool_name", "arguments": {"arg1": value}}

Do not include any text outside the JSON response."""


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

        return f"""Goal: {self.state.goal}

Step: {self.state.step_count}

Available tools:
{tools_desc}

Current observation:
{json.dumps(observation, indent=2)}

{f"Last action: {json.dumps(self.state.last_action)}" if self.state.last_action else ""}

Analyze the situation and respond with a JSON action."""

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

