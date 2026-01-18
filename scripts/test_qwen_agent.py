#!/usr/bin/env python3
"""Quick test for Qwen agent with MCP.

Usage:
    # First, start the MCP server:
    python -m mcp_server.server
    
    # Then run this test:
    python scripts/test_qwen_agent.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.base_agent import AgentConfig
from agents.qwen_agent import QwenAgent


async def test_qwen_basic():
    """Test basic Qwen agent functionality."""
    print("=" * 60)
    print("Testing Qwen Agent (qwen2.5:3b via Ollama)")
    print("=" * 60)
    
    # Check Ollama is running
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code != 200:
                print("ERROR: Ollama not running. Start with: ollama serve")
                return False
            
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"Available Ollama models: {models}")
            
            if "qwen2.5:3b" not in models:
                print("WARNING: qwen2.5:3b not found. Pull with: ollama pull qwen2.5:3b")
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama: {e}")
        return False
    
    # Create agent
    config = AgentConfig(
        mcp_server_url="http://localhost:8080",
        max_steps=5,
        name="qwen_test",
    )
    
    agent = QwenAgent(config=config, model="qwen2.5:3b")
    print(f"\nAgent: {agent.model_name}")
    
    # Test with a simple goal
    goal = "Get the current world graph and report what spatial predicates are detected"
    print(f"Goal: {goal}\n")
    
    try:
        step_count = 0
        async for step in agent.run(goal):
            step_count += 1
            status = step.get("status", "unknown")
            print(f"Step {step_count}: {status}")
            
            if status == "executed":
                action = step.get("action", {})
                print(f"  Tool: {action.get('tool', 'N/A')}")
                result = step.get("result", {})
                if isinstance(result, dict):
                    if "predicates" in result:
                        print(f"  Predicates found: {len(result['predicates'])}")
                    elif "world_context" in result:
                        ctx = result["world_context"]
                        print(f"  Graph: {ctx.get('num_nodes', '?')} nodes, {ctx.get('num_edges', '?')} edges")
            
            elif status == "complete":
                print(f"  Reason: {step.get('reason', 'N/A')}")
                print("\n✓ Qwen agent completed successfully!")
                return True
            
            elif status == "error":
                print(f"  Error: {step.get('error', 'Unknown')}")
                return False
        
        print("\n✓ Test finished")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        return False
    finally:
        await agent.close()


async def test_qwen_json_output():
    """Test Qwen's JSON output formatting."""
    print("\n" + "=" * 60)
    print("Testing Qwen JSON Output")
    print("=" * 60)
    
    import httpx
    
    test_prompt = """Current Goal: Report detected predicates
Step: 1/5

Current State:
- Spatial predicates detected: 45
- World graph nodes: 16

Since spatial_count > 0, the task is achieved. Return complete action.

JSON only:"""

    system = """You control a robot via MCP.

CRITICAL: Respond with EXACTLY one of these JSON formats:

FORMAT 1 - Complete: {"action": "complete", "reason": "what you found"}
FORMAT 2 - Tool call: {"action": "tool_call", "tool": "get_world_graph", "arguments": {}}

RULES:
- If spatial predicates > 0: Return complete action
- Otherwise: Call get_world_graph tool

JSON ONLY."""

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen2.5:3b",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": test_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            },
        )
        
        if response.status_code != 200:
            print(f"ERROR: Ollama request failed: {response.status_code}")
            return False
        
        result = response.json()
        content = result.get("message", {}).get("content", "")
        
        print(f"Raw response:\n{content}\n")
        
        # Try to parse JSON
        import json
        try:
            # Handle markdown code blocks
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                parsed = json.loads(content[json_start:json_end])
                print(f"Parsed JSON: {parsed}")
                print("\n✓ JSON parsing successful!")
                return True
            else:
                print("✗ No JSON found in response")
                return False
                
        except json.JSONDecodeError as e:
            print(f"✗ JSON parse error: {e}")
            return False


async def main():
    """Run all tests."""
    # Test JSON output first (no MCP server needed)
    json_ok = await test_qwen_json_output()
    
    if not json_ok:
        print("\nJSON test failed. Fix Qwen response format before MCP test.")
        return
    
    # Full MCP test (requires server running)
    print("\nNote: Full MCP test requires server running: python -m mcp_server.server")
    
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:8080/health", timeout=2.0)
            if resp.status_code == 200:
                await test_qwen_basic()
            else:
                print("MCP server not responding. Skipping full test.")
    except Exception:
        print("MCP server not running. Skipping full test.")
        print("Start with: cd src && python -m mcp_server.server")


if __name__ == "__main__":
    asyncio.run(main())
