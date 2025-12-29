#!/usr/bin/env python3
"""Run MCP agent experiments for thesis validation.

This script runs the "swappable brain" experiment:
1. Launch simulation with TurtleBot3
2. Start MCP-ROS2 bridge
3. Run navigation task with Claude agent
4. Run same task with Llama agent
5. Compare results and log for thesis
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.base_agent import AgentConfig
from agents.claude_agent import ClaudeAgent
from agents.llama_agent import LlamaAgent


async def run_agent_experiment(
    agent_type: str,
    goal: str,
    mcp_url: str,
    max_steps: int = 30,
) -> dict:
    """Run experiment with specified agent.
    
    Args:
        agent_type: "claude" or "llama"
        goal: Navigation goal description
        mcp_url: MCP server URL
        max_steps: Maximum steps before timeout
        
    Returns:
        Experiment results dictionary
    """
    config = AgentConfig(
        mcp_server_url=mcp_url,
        max_steps=max_steps,
        name=f"{agent_type}_experiment",
    )

    if agent_type == "claude":
        agent = ClaudeAgent(config=config)
    elif agent_type == "llama":
        agent = LlamaAgent(config=config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    results = {
        "agent": agent.model_name,
        "goal": goal,
        "start_time": datetime.now().isoformat(),
        "steps": [],
        "success": False,
        "total_steps": 0,
        "duration_seconds": 0,
    }

    start_time = time.time()

    try:
        async for step in agent.run(goal):
            results["steps"].append(step)
            print(f"[{agent_type}] Step {step.get('step', '?')}: {step.get('status', 'unknown')}")
            
            if step.get("status") == "complete":
                results["success"] = True
            elif step.get("status") == "error":
                results["error"] = step.get("error")

    except Exception as e:
        results["error"] = str(e)

    results["duration_seconds"] = round(time.time() - start_time, 2)
    results["total_steps"] = len([s for s in results["steps"] if s.get("status") == "executed"])
    results["end_time"] = datetime.now().isoformat()

    return results


async def run_comparison_experiment(
    goal: str,
    mcp_url: str,
    output_dir: Path,
) -> None:
    """Run same task with multiple agents and compare.
    
    This is the key thesis validation: demonstrate that different
    AI models can control the same robot through MCP without
    any robot-side code changes.
    """
    print("=" * 60)
    print("MCP Swappable Brain Experiment")
    print("=" * 60)
    print(f"Goal: {goal}")
    print(f"MCP Server: {mcp_url}")
    print("=" * 60)

    all_results = {
        "experiment_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "goal": goal,
        "mcp_url": mcp_url,
        "agents": {},
    }

    # Run with Claude
    print("\n[1/2] Running with Claude agent...")
    try:
        claude_results = await run_agent_experiment("claude", goal, mcp_url)
        all_results["agents"]["claude"] = claude_results
        print(f"Claude: {'SUCCESS' if claude_results['success'] else 'FAILED'} "
              f"in {claude_results['total_steps']} steps ({claude_results['duration_seconds']}s)")
    except Exception as e:
        print(f"Claude experiment failed: {e}")
        all_results["agents"]["claude"] = {"error": str(e)}

    # Brief pause between experiments
    await asyncio.sleep(2)

    # Run with Llama
    print("\n[2/2] Running with Llama agent...")
    try:
        llama_results = await run_agent_experiment("llama", goal, mcp_url)
        all_results["agents"]["llama"] = llama_results
        print(f"Llama: {'SUCCESS' if llama_results['success'] else 'FAILED'} "
              f"in {llama_results['total_steps']} steps ({llama_results['duration_seconds']}s)")
    except Exception as e:
        print(f"Llama experiment failed: {e}")
        all_results["agents"]["llama"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    for agent_name, results in all_results["agents"].items():
        if "error" in results:
            print(f"{agent_name}: ERROR - {results['error']}")
        else:
            status = "✓ SUCCESS" if results.get("success") else "✗ FAILED"
            print(f"{agent_name}: {status}")
            print(f"  Steps: {results.get('total_steps', 'N/A')}")
            print(f"  Duration: {results.get('duration_seconds', 'N/A')}s")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"experiment_{all_results['experiment_id']}.json"
    
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MCP agent experiments for thesis validation"
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="Navigate to the goal marker while avoiding obstacles",
        help="Navigation goal description",
    )
    parser.add_argument(
        "--mcp-url",
        type=str,
        default="http://localhost:8080",
        help="MCP server URL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["claude", "llama", "both"],
        default="both",
        help="Which agent(s) to run",
    )

    args = parser.parse_args()

    if args.agent == "both":
        asyncio.run(run_comparison_experiment(
            goal=args.goal,
            mcp_url=args.mcp_url,
            output_dir=args.output_dir,
        ))
    else:
        result = asyncio.run(run_agent_experiment(
            agent_type=args.agent,
            goal=args.goal,
            mcp_url=args.mcp_url,
        ))
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

