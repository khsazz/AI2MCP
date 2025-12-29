# AI2MCP: A Standardized Middleware Architecture for Decoupled Robotic Intelligence

[![ROS 2](https://img.shields.io/badge/ROS%202-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-1.0-purple)](https://modelcontextprotocol.io/)

> **Thesis Project**: Demonstrating that robotic intelligence can be treated as a swappable service using the Model Context Protocol (MCP).

## Overview

This project implements an **MCP-to-ROS 2 Bridge** that allows any AI model (Claude, Llama, GPT, etc.) to control robots through a standardized protocol. The key innovation is treating the AI "brain" as a swappable component—similar to how USB-C standardizes hardware connections.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLOUD / WORKSTATION                          │
│  ┌───────────────┐    ┌───────────────────┐    ┌─────────────────┐  │
│  │   LLM Agent   │◄──►│   MCP Client SDK  │◄──►│  GNN Reasoner   │  │
│  │ (Claude/Llama)│    │   (JSON-RPC/SSE)  │    │ (World Graph)   │  │
│  └───────────────┘    └─────────┬─────────┘    └─────────────────┘  │
└─────────────────────────────────┼───────────────────────────────────┘
                                  │ HTTP/SSE
┌─────────────────────────────────┼───────────────────────────────────┐
│                        ROBOT / EDGE (ROS 2 Humble)                  │
│                     ┌───────────▼─────────┐                         │
│                     │   MCP-ROS2 Bridge   │                         │
│                     │   (Tools/Resources) │                         │
│                     └───────────┬─────────┘                         │
│                                 │                                   │
│           ┌─────────────────────┼─────────────────────┐             │
│           ▼                     ▼                     ▼             │
│       /cmd_vel              /scan                /camera            │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

- **🔌 Protocol-Driven**: Standardized MCP interface for AI-robot communication
- **🔄 Swappable AI**: Change the "brain" (Claude ↔ Llama) without modifying robot code
- **🧠 Semantic Perception**: GNN-based world graph for structured environment understanding
- **📊 Explainability**: All AI decisions logged as tool calls and resource queries
- **☁️ Cloud-Edge Split**: Heavy compute on cloud, lightweight execution on robot

## Quick Start

### Prerequisites

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10+
- (Optional) Gazebo Classic for simulation

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/AI2MCP.git
cd AI2MCP

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Source ROS 2
source /opt/ros/humble/setup.bash
```

### Running the Demo

**Terminal 1: Start Gazebo Simulation**
```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

**Terminal 2: Start MCP-ROS2 Bridge**
```bash
source /opt/ros/humble/setup.bash
source .venv/bin/activate
./scripts/start_bridge.sh
```

**Terminal 3: Run AI Agent**
```bash
source .venv/bin/activate

# With Claude (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY="your-key"
python scripts/run_experiment.py --agent claude --goal "Navigate to position (3, -3)"

# With Llama (requires Ollama running locally)
python scripts/run_experiment.py --agent llama --goal "Navigate to position (3, -3)"
```

## Project Structure

```
AI2MCP/
├── src/
│   ├── mcp_ros2_bridge/      # MCP server with ROS 2 integration
│   │   ├── server.py         # SSE-based MCP server
│   │   ├── ros_node.py       # ROS 2 node wrapper
│   │   ├── tools/            # MCP tools (motion, perception)
│   │   └── resources/        # MCP resources (pose, scan, world_graph)
│   │
│   ├── gnn_reasoner/         # Semantic scene understanding
│   │   ├── detector.py       # YOLO-based object detection
│   │   ├── graph_builder.py  # Sensor → World Graph
│   │   ├── model/            # PyTorch Geometric GNN
│   │   └── service.py        # HTTP inference service
│   │
│   └── agents/               # Swappable AI agents
│       ├── base_agent.py     # Abstract agent interface
│       ├── claude_agent.py   # Anthropic Claude implementation
│       └── llama_agent.py    # Local Llama (Ollama/vLLM)
│
├── simulation/               # Gazebo simulation setup
│   ├── launch/              # ROS 2 launch files
│   ├── config/              # Nav2 parameters
│   └── worlds/              # Custom Gazebo worlds
│
├── tests/                   # Test suite
├── scripts/                 # Utility scripts
└── docs/                    # Documentation
```

## MCP Interface

### Tools (Actions)

| Tool | Description |
|------|-------------|
| `move(linear_x, angular_z, duration_ms)` | Move with velocity for duration |
| `stop()` | Emergency stop |
| `rotate(angle_degrees)` | Rotate in place |
| `move_forward(distance_meters)` | Move forward by distance |
| `get_obstacle_distances(directions)` | Query obstacles in directions |
| `check_path_clear(distance, width)` | Check if path is navigable |
| `scan_surroundings(num_sectors)` | 360° obstacle scan |

### Resources (State)

| Resource URI | Description |
|--------------|-------------|
| `robot://pose` | Current position (x, y, θ) |
| `robot://velocity` | Current velocity |
| `robot://scan/summary` | LiDAR scan summary by quadrant |
| `robot://scan/obstacles` | Detected obstacle clusters |
| `robot://world_graph` | Semantic scene graph |

## Thesis Validation

The key experiment demonstrates **swappable intelligence**:

1. Run navigation task with Claude agent
2. Run identical task with Llama agent  
3. **No robot-side code changes** between runs

```bash
# Run comparison experiment
python scripts/run_experiment.py --agent both --goal "Explore the room"
```

Results are saved to `experiments/results/` for thesis analysis.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SERVER_HOST` | `0.0.0.0` | MCP server bind address |
| `MCP_SERVER_PORT` | `8080` | MCP server port |
| `ANTHROPIC_API_KEY` | - | Claude API key |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |

### Agent Configuration

```python
from agents.base_agent import AgentConfig
from agents.claude_agent import ClaudeAgent

config = AgentConfig(
    mcp_server_url="http://localhost:8080",
    max_steps=50,
    timeout_seconds=30.0,
)

agent = ClaudeAgent(config=config, model="claude-sonnet-4-20250514")
```

## Development

```bash
# Run tests
pytest tests/ -v

# Type checking
mypy src/

# Linting
ruff check src/

# Format
ruff format src/
```

## Research Contributions

1. **N×M → N+M Complexity**: Single MCP interface per robot connects to any model
2. **Explainable Robot AI**: All decisions logged as structured tool calls
3. **Semantic Perception**: GNN-processed world graphs for structured reasoning
4. **Protocol-Driven Robotics**: Foundation for multi-robot, multi-agent systems

## Citation

```bibtex
@mastersthesis{sazzad2025mcp,
  title={A Standardized Middleware Architecture for Decoupled Robotic 
         Intelligence using the Model Context Protocol},
  author={Sazzad, Khaled},
  year={2025},
  school={Your University}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Model Context Protocol](https://modelcontextprotocol.io/) by Anthropic
- [ROS 2](https://ros.org/) by Open Robotics
- [PyTorch Geometric](https://pyg.org/) for GNN implementation

