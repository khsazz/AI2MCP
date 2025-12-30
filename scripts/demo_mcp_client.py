#!/usr/bin/env python3
"""MCP Client Demo - End-to-End Protocol Demonstration.

Connects to the MCP-ROS2 bridge server and demonstrates:
1. Tool discovery (list available capabilities)
2. Resource access (world graph, predicates)
3. Prediction tool calls (get_world_graph, predict_action_outcome)

Usage:
    # First, start the server in another terminal:
    python -m mcp_ros2_bridge.server --lerobot

    # Then run this client:
    python scripts/demo_mcp_client.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp import ClientSession
from mcp.client.sse import sse_client


async def run_demo(base_url: str = "http://localhost:8080") -> None:
    """Run the full MCP E2E demonstration."""
    print("=" * 70)
    print("MCP-ROS2 Bridge End-to-End Demo")
    print("=" * 70)
    print(f"Server: {base_url}")
    print()

    sse_url = f"{base_url}/sse"

    try:
        async with sse_client(sse_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # Step 1: Initialize
                print("[1/5] Initializing MCP Session...")
                await session.initialize()
                print("      ✓ Session initialized")

                # Step 2: List tools
                print("\n[2/5] Discovering Tools...")
                tools_result = await session.list_tools()
                tools = tools_result.tools
                print(f"      Found {len(tools)} tools:")
                for tool in tools[:8]:  # Show first 8
                    desc = (tool.description or "")[:50]
                    print(f"        - {tool.name}: {desc}...")
                if len(tools) > 8:
                    print(f"        ... and {len(tools) - 8} more")

                # Step 3: List resources
                print("\n[3/5] Discovering Resources...")
                resources_result = await session.list_resources()
                resources = resources_result.resources
                print(f"      Found {len(resources)} resources:")
                for res in resources[:8]:  # Show first 8
                    print(f"        - {res.uri}")
                if len(resources) > 8:
                    print(f"        ... and {len(resources) - 8} more")

                # Step 4: Read a resource
                print("\n[4/5] Reading LeRobot World Graph Resource...")
                try:
                    start = time.perf_counter()
                    resource_result = await session.read_resource("robot://lerobot/world_graph")
                    latency = (time.perf_counter() - start) * 1000

                    for content in resource_result.contents:
                        text = getattr(content, 'text', None)
                        if text:
                            data = json.loads(text)
                            print(f"      Nodes: {data.get('num_nodes', 'N/A')}")
                            print(f"      Edges: {data.get('num_edges', 'N/A')}")
                            spatial = data.get('spatial_predicates', [])
                            interaction = data.get('interaction_predicates', [])
                            print(f"      Spatial predicates: {len(spatial)}")
                            print(f"      Interaction predicates: {len(interaction)}")

                            # Show sample predicates
                            if spatial[:3]:
                                print("      Sample spatial predicates:")
                                for p in spatial[:3]:
                                    print(f"        {p['source']} --{p['predicate']}--> {p['target']} ({p['confidence']:.2f})")

                    print(f"      ✓ Resource read in {latency:.1f}ms")
                except Exception as e:
                    print(f"      ✗ Failed to read resource: {e}")

                # Step 5: Call a tool
                print("\n[5/5] Calling get_world_graph Tool...")
                try:
                    start = time.perf_counter()
                    tool_result = await session.call_tool("get_world_graph", {"threshold": 0.3})
                    latency = (time.perf_counter() - start) * 1000

                    for content in tool_result.content:
                        text = getattr(content, 'text', None)
                        if text:
                            try:
                                data = json.loads(text)
                                print(f"      Frame index: {data.get('frame_index', 'N/A')}")
                                ctx = data.get('world_context', {})
                                print(f"      Graph nodes: {ctx.get('num_nodes', 'N/A')}")
                                inf_time = data.get('inference_time_ms')
                                if inf_time:
                                    print(f"      Inference time: {inf_time:.2f}ms")
                            except json.JSONDecodeError:
                                print(f"      Raw response: {text[:100]}...")

                    print(f"      ✓ Tool call completed in {latency:.1f}ms")
                except Exception as e:
                    print(f"      ✗ Failed to call tool: {e}")

                # Bonus: Benchmark multiple calls
                print("\n[Bonus] Benchmarking Tool Calls...")
                try:
                    latencies = []
                    for i in range(10):
                        start = time.perf_counter()
                        await session.call_tool("advance_frame")
                        await session.call_tool("get_world_graph", {"threshold": 0.3})
                        latencies.append((time.perf_counter() - start) * 1000)

                    avg = sum(latencies) / len(latencies)
                    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
                    print(f"      10 sequential advance+predict cycles:")
                    print(f"      Avg latency: {avg:.1f}ms")
                    print(f"      P95 latency: {p95:.1f}ms")
                    print(f"      Throughput: {1000/avg:.1f} ops/sec")
                except Exception as e:
                    print(f"      ✗ Benchmark failed: {e}")

    except Exception as e:
        print(f"      ✗ Connection failed: {e}")
        print("\n      Make sure the server is running:")
        print("      python -m mcp_ros2_bridge.server --lerobot")
        return

    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="MCP Client E2E Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080",
        help="MCP server URL",
    )
    args = parser.parse_args()

    print("\nStarting MCP Client Demo...")
    print("Make sure the server is running first:\n")
    print("  python -m mcp_ros2_bridge.server --lerobot\n")

    asyncio.run(run_demo(args.url))


if __name__ == "__main__":
    main()
