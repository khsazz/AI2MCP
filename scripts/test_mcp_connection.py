#!/usr/bin/env python3
"""Test MCP server connection and basic functionality.

Quick diagnostic script to verify the MCP-ROS2 bridge is working.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def test_connection(url: str = "http://localhost:8080") -> None:
    """Test MCP server connectivity."""
    print(f"Testing MCP server at: {url}")
    print("=" * 50)

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test health endpoint
        print("\n1. Health Check...")
        try:
            response = await client.get(f"{url}/health")
            if response.status_code == 200:
                print(f"   ✓ Server healthy: {response.json()}")
            else:
                print(f"   ✗ Unexpected status: {response.status_code}")
        except Exception as e:
            print(f"   ✗ Connection failed: {e}")
            return

        # Test SSE endpoint exists
        print("\n2. SSE Endpoint...")
        try:
            # Just check if endpoint responds (don't actually connect SSE)
            response = await client.get(f"{url}/sse", timeout=2.0)
            print(f"   ✓ SSE endpoint available (status: {response.status_code})")
        except httpx.ReadTimeout:
            print("   ✓ SSE endpoint available (streaming)")
        except Exception as e:
            print(f"   ? SSE check: {e}")

        # Test message endpoint
        print("\n3. Message Endpoint...")
        try:
            response = await client.post(
                f"{url}/messages/",
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"}
                    },
                    "id": 1
                }
            )
            print(f"   Response: {response.status_code}")
            if response.text:
                print(f"   Body: {response.text[:200]}...")
        except Exception as e:
            print(f"   ? Message endpoint: {e}")

    print("\n" + "=" * 50)
    print("Connection test complete")


def main() -> None:
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MCP server connection")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080",
        help="MCP server URL"
    )
    
    args = parser.parse_args()
    asyncio.run(test_connection(args.url))


if __name__ == "__main__":
    main()

