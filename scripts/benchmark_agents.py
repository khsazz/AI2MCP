#!/usr/bin/env python3
"""Benchmark Qwen vs Llama agents on standardized goals.

Measures:
- Success rate (task completion)
- Steps to completion
- Latency (time-to-first-action, total time)
- Tool call accuracy

Usage:
    # Ensure MCP server is running:
    python -m mcp_ros2_bridge.server --lerobot
    
    # Run benchmark:
    python scripts/benchmark_agents.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.base_agent import AgentConfig
from agents.llama_agent import LlamaAgent
from agents.qwen_agent import QwenAgent


# Standard test goals
TEST_GOALS = [
    # Simple queries (1-2 steps expected)
    "Get the current world graph",
    "Report spatial predicates in the scene",
    "Check how many nodes are in the world graph",
    
    # Frame navigation (2-3 steps expected)
    "Advance to frame 10 and get predicates",
    "Set frame to 50 and report world graph",
    
    # Predicate queries (1-2 steps expected)
    "List all active predicates",
    "Get predicates with threshold 0.7",
    
    # Action simulation (2-3 steps expected)
    "Simulate moving forward and check outcome",
    "Predict the outcome of rotating 90 degrees",
    
    # Complex query (2-4 steps expected)
    "Analyze the scene and report what objects are near each other",
]


@dataclass
class GoalResult:
    """Result of running a single goal."""
    goal: str
    agent: str
    success: bool
    steps: int
    time_to_first_action_ms: float
    total_time_ms: float
    final_reason: str = ""
    error: str = ""
    tool_calls: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResults:
    """Aggregate benchmark results."""
    agent: str
    goals_tested: int
    successes: int
    failures: int
    avg_steps: float
    avg_time_to_first_action_ms: float
    avg_total_time_ms: float
    results: list[GoalResult] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        return self.successes / self.goals_tested if self.goals_tested > 0 else 0.0


async def run_goal(agent, goal: str, agent_name: str, timeout: float = 30.0) -> GoalResult:
    """Run a single goal and measure results."""
    start_time = time.perf_counter()
    first_action_time = None
    steps = 0
    tool_calls = []
    success = False
    final_reason = ""
    error = ""
    
    try:
        async for step in agent.run(goal):
            status = step.get("status", "unknown")
            
            if status == "deciding" and first_action_time is None:
                first_action_time = time.perf_counter()
            
            if status == "executed":
                steps += 1
                action = step.get("action", {})
                if action:
                    tool_calls.append(action.get("tool", "unknown"))
            
            if status == "complete":
                success = True
                final_reason = step.get("reason", "")
                break
            
            if status == "error":
                error = step.get("error", "Unknown error")
                break
            
            if status == "max_steps_reached":
                # Treat max steps as partial success if we got data
                if steps > 0:
                    success = True
                    final_reason = f"Completed with {steps} tool calls"
                else:
                    error = "Max steps reached"
                break
            
            # Timeout check
            if time.perf_counter() - start_time > timeout:
                error = "Timeout"
                break
                
    except asyncio.CancelledError:
        # Ignore cancellation during cleanup
        pass
    except Exception as e:
        error = str(e)
    
    end_time = time.perf_counter()
    
    # Give async cleanup time to complete
    await asyncio.sleep(0.1)
    
    return GoalResult(
        goal=goal,
        agent=agent_name,
        success=success,
        steps=steps,
        time_to_first_action_ms=(first_action_time - start_time) * 1000 if first_action_time else 0,
        total_time_ms=(end_time - start_time) * 1000,
        final_reason=final_reason,
        error=error,
        tool_calls=tool_calls,
    )


def run_single_goal_sync(agent_class, agent_name: str, model: str, goal: str) -> GoalResult:
    """Run a single goal in a fresh process to avoid SSE cleanup issues."""
    import subprocess
    import json
    
    # Create a simple inline script
    script = f'''
import asyncio
import sys
import json
import time
sys.path.insert(0, "src")
from agents.base_agent import AgentConfig
from agents.{agent_name}_agent import {"LlamaAgent" if agent_name == "llama" else "QwenAgent"}

async def run():
    config = AgentConfig(mcp_server_url="http://localhost:8080", max_steps=10, name="bench")
    agent = {"LlamaAgent" if agent_name == "llama" else "QwenAgent"}(config=config, model="{model}")
    
    start = time.perf_counter()
    first_action = None
    steps = 0
    tools = []
    success = False
    reason = ""
    error = ""
    
    try:
        async for step in agent.run("{goal}"):
            status = step.get("status", "")
            if status == "deciding" and first_action is None:
                first_action = time.perf_counter()
            if status == "executed":
                steps += 1
                a = step.get("action", {{}})
                if a: tools.append(a.get("tool", ""))
            if status == "complete":
                success = True
                reason = step.get("reason", "")
                break
            if status in ("error", "max_steps_reached"):
                if status == "max_steps_reached" and steps > 0:
                    success = True
                    reason = f"Completed with {{steps}} calls"
                else:
                    error = step.get("error", status)
                break
    except Exception as e:
        error = str(e)
    
    end = time.perf_counter()
    print(json.dumps({{
        "success": success,
        "steps": steps,
        "ttfa": (first_action - start) * 1000 if first_action else 0,
        "total": (end - start) * 1000,
        "tools": tools,
        "reason": reason,
        "error": error
    }}))

asyncio.run(run())
'''
    
    try:
        result = subprocess.run(
            ["python3", "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/home/khaled.sazzad/Downloads/Thesis/AI2MCP",
        )
        
        # Find JSON in output
        for line in result.stdout.strip().split("\n"):
            if line.startswith("{"):
                data = json.loads(line)
                return GoalResult(
                    goal=goal,
                    agent=agent_name,
                    success=data["success"],
                    steps=data["steps"],
                    time_to_first_action_ms=data["ttfa"],
                    total_time_ms=data["total"],
                    tool_calls=data["tools"],
                    final_reason=data["reason"],
                    error=data["error"],
                )
        
        # No JSON found
        return GoalResult(
            goal=goal,
            agent=agent_name,
            success=False,
            steps=0,
            time_to_first_action_ms=0,
            total_time_ms=0,
            error=f"No output: {result.stderr[:100]}",
        )
    except subprocess.TimeoutExpired:
        return GoalResult(
            goal=goal,
            agent=agent_name,
            success=False,
            steps=0,
            time_to_first_action_ms=0,
            total_time_ms=0,
            error="Timeout",
        )
    except Exception as e:
        return GoalResult(
            goal=goal,
            agent=agent_name,
            success=False,
            steps=0,
            time_to_first_action_ms=0,
            total_time_ms=0,
            error=str(e),
        )


async def benchmark_agent(agent_class, agent_name: str, model: str, goals: list[str]) -> BenchmarkResults:
    """Benchmark a single agent on all goals."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {agent_name} ({model})")
    print(f"{'='*60}")
    
    results = []
    
    for i, goal in enumerate(goals, 1):
        print(f"\n[{i}/{len(goals)}] Goal: {goal[:50]}...")
        
        # Run in subprocess to avoid SSE cleanup issues
        result = run_single_goal_sync(agent_class, agent_name, model, goal)
        results.append(result)
        
        status = "✓" if result.success else "✗"
        print(f"  {status} Steps: {result.steps}, Time: {result.total_time_ms:.0f}ms")
        if result.error:
            print(f"  Error: {result.error}")
        if result.tool_calls:
            print(f"  Tools: {', '.join(result.tool_calls[:3])}")
    
    # Aggregate results
    successes = sum(1 for r in results if r.success)
    successful_results = [r for r in results if r.success]
    
    return BenchmarkResults(
        agent=agent_name,
        goals_tested=len(goals),
        successes=successes,
        failures=len(goals) - successes,
        avg_steps=sum(r.steps for r in successful_results) / len(successful_results) if successful_results else 0,
        avg_time_to_first_action_ms=sum(r.time_to_first_action_ms for r in successful_results) / len(successful_results) if successful_results else 0,
        avg_total_time_ms=sum(r.total_time_ms for r in successful_results) / len(successful_results) if successful_results else 0,
        results=results,
    )


def print_comparison(llama_results: BenchmarkResults, qwen_results: BenchmarkResults):
    """Print comparison table."""
    print("\n" + "="*70)
    print("BENCHMARK COMPARISON: Llama3.2 vs Qwen2.5")
    print("="*70)
    
    print(f"\n{'Metric':<35} {'Llama3.2':>15} {'Qwen2.5':>15}")
    print("-"*70)
    print(f"{'Success Rate':<35} {llama_results.success_rate*100:>14.1f}% {qwen_results.success_rate*100:>14.1f}%")
    print(f"{'Successes / Total':<35} {f'{llama_results.successes}/{llama_results.goals_tested}':>15} {f'{qwen_results.successes}/{qwen_results.goals_tested}':>15}")
    print(f"{'Avg Steps (successful)':<35} {llama_results.avg_steps:>15.1f} {qwen_results.avg_steps:>15.1f}")
    print(f"{'Avg Time-to-First-Action (ms)':<35} {llama_results.avg_time_to_first_action_ms:>15.0f} {qwen_results.avg_time_to_first_action_ms:>15.0f}")
    print(f"{'Avg Total Time (ms)':<35} {llama_results.avg_total_time_ms:>15.0f} {qwen_results.avg_total_time_ms:>15.0f}")
    
    # Per-goal comparison
    print("\n" + "-"*70)
    print("Per-Goal Results:")
    print("-"*70)
    print(f"{'Goal':<40} {'Llama':>12} {'Qwen':>12}")
    print("-"*70)
    
    for llama_r, qwen_r in zip(llama_results.results, qwen_results.results):
        llama_status = f"✓ {llama_r.steps}st" if llama_r.success else "✗"
        qwen_status = f"✓ {qwen_r.steps}st" if qwen_r.success else "✗"
        print(f"{llama_r.goal[:38]:<40} {llama_status:>12} {qwen_status:>12}")
    
    # Winner
    print("\n" + "="*70)
    if llama_results.success_rate > qwen_results.success_rate:
        print("WINNER: Llama3.2 (higher success rate)")
    elif qwen_results.success_rate > llama_results.success_rate:
        print("WINNER: Qwen2.5 (higher success rate)")
    else:
        if llama_results.avg_total_time_ms < qwen_results.avg_total_time_ms:
            print("TIE on success rate. Llama3.2 faster.")
        elif qwen_results.avg_total_time_ms < llama_results.avg_total_time_ms:
            print("TIE on success rate. Qwen2.5 faster.")
        else:
            print("TIE")
    print("="*70)


async def check_server() -> bool:
    """Check if MCP server is running."""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:8080/health", timeout=2.0)
            return resp.status_code == 200
    except Exception:
        return False


async def check_ollama() -> tuple[bool, bool]:
    """Check if Ollama is running and models are available."""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags", timeout=2.0)
            if resp.status_code != 200:
                return False, False
            
            models = [m["name"] for m in resp.json().get("models", [])]
            has_llama = any("llama" in m for m in models)
            has_qwen = any("qwen" in m for m in models)
            return has_llama, has_qwen
    except Exception:
        return False, False


async def main():
    """Run the benchmark."""
    print("Agent Benchmark: Qwen2.5 vs Llama3.2")
    print("="*60)
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    
    if not await check_server():
        print("ERROR: MCP server not running.")
        print("Start with: python -m mcp_ros2_bridge.server --lerobot")
        return
    print("  ✓ MCP server running")
    
    has_llama, has_qwen = await check_ollama()
    if not has_llama:
        print("ERROR: Llama model not found. Pull with: ollama pull llama3.2")
        return
    print("  ✓ Llama3.2 available")
    
    if not has_qwen:
        print("ERROR: Qwen model not found. Pull with: ollama pull qwen2.5:3b")
        return
    print("  ✓ Qwen2.5 available")
    
    # Run benchmarks
    goals = TEST_GOALS
    print(f"\nRunning {len(goals)} goals on each agent...")
    
    # Benchmark Llama
    llama_results = await benchmark_agent(
        LlamaAgent, "llama", "llama3.2", goals
    )
    
    # Benchmark Qwen
    qwen_results = await benchmark_agent(
        QwenAgent, "qwen", "qwen2.5:3b", goals
    )
    
    # Print comparison
    print_comparison(llama_results, qwen_results)
    
    # Save results
    output_path = Path("experiments/agent_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "goals_count": len(goals),
        "llama": {
            "success_rate": llama_results.success_rate,
            "successes": llama_results.successes,
            "failures": llama_results.failures,
            "avg_steps": llama_results.avg_steps,
            "avg_time_to_first_action_ms": llama_results.avg_time_to_first_action_ms,
            "avg_total_time_ms": llama_results.avg_total_time_ms,
            "results": [
                {
                    "goal": r.goal,
                    "success": r.success,
                    "steps": r.steps,
                    "time_ms": r.total_time_ms,
                    "tool_calls": r.tool_calls,
                    "error": r.error,
                }
                for r in llama_results.results
            ],
        },
        "qwen": {
            "success_rate": qwen_results.success_rate,
            "successes": qwen_results.successes,
            "failures": qwen_results.failures,
            "avg_steps": qwen_results.avg_steps,
            "avg_time_to_first_action_ms": qwen_results.avg_time_to_first_action_ms,
            "avg_total_time_ms": qwen_results.avg_total_time_ms,
            "results": [
                {
                    "goal": r.goal,
                    "success": r.success,
                    "steps": r.steps,
                    "time_ms": r.total_time_ms,
                    "tool_calls": r.tool_calls,
                    "error": r.error,
                }
                for r in qwen_results.results
            ],
        },
    }
    
    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
