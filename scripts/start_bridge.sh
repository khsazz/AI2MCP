#!/bin/bash
# Start MCP-ROS2 Bridge Server
#
# This script starts the MCP server that bridges AI agents to ROS 2.
# Ensure ROS 2 Humble is sourced before running.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting MCP-ROS2 Bridge${NC}"
echo "================================"

# Check ROS 2 environment
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${YELLOW}Warning: ROS 2 not sourced. Attempting to source Humble...${NC}"
    if [ -f "/opt/ros/humble/setup.bash" ]; then
        source /opt/ros/humble/setup.bash
        echo -e "${GREEN}ROS 2 Humble sourced successfully${NC}"
    else
        echo -e "${RED}Error: ROS 2 Humble not found. Please install and source ROS 2.${NC}"
        exit 1
    fi
fi

echo "ROS_DISTRO: $ROS_DISTRO"

# Check Python environment
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv "$PROJECT_ROOT/.venv"
    source "$PROJECT_ROOT/.venv/bin/activate"
    pip install -e "$PROJECT_ROOT[dev]"
else
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

echo "Python: $(which python3)"
echo "================================"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export MCP_SERVER_HOST="${MCP_SERVER_HOST:-0.0.0.0}"
export MCP_SERVER_PORT="${MCP_SERVER_PORT:-8080}"

echo "MCP Server: http://$MCP_SERVER_HOST:$MCP_SERVER_PORT"
echo "================================"

# Start the server
cd "$PROJECT_ROOT"
python3 -m mcp_ros2_bridge.server

