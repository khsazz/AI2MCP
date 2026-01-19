# AI2MCP: A Standardized Middleware Architecture for Decoupled Robotic Intelligence

[![ROS 2](https://img.shields.io/badge/ROS%202-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-1.0-purple)](https://modelcontextprotocol.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)
[![LeRobot](https://img.shields.io/badge/LeRobot-HuggingFace-yellow)](https://huggingface.co/lerobot)

> **Thesis Project**: Demonstrating that robotic intelligence can be treated as a swappable service using the Model Context Protocol (MCP).

## Key Results (Fair Comparison, 55k frames)

| Metric | RelationalGNN | MultiModalGNN | Winner |
|--------|---------------|---------------|--------|
| **Accuracy** | **97.03%** | 96.51% | ✅ Kinematic |
| **`is_near` F1** | **0.954** | 0.920 | ✅ Kinematic |
| **Latency** | **1.5ms** | 24ms | ✅ Kinematic (16×) |
| **Model Size** | **0.81MB** | 2.14MB | ✅ Kinematic (2.6×) |
| **Pass@1** | 88.2% | — | — |

### LLM Agent Benchmark

| Metric | Llama3.2 (3B) | Qwen2.5 (3B) | Winner |
|--------|---------------|--------------|--------|
| **Success Rate** | 100% | 100% | TIE |
| **Avg Steps** | 2.8 | **1.0** | ✅ Qwen |
| **Time-to-First-Action** | **425ms** | 1073ms | ✅ Llama |
| **Avg Total Time** | 2561ms | **1537ms** | ✅ Qwen (40% faster) |

> ⚠️ **Finding**: RelationalGNN outperforms MultiModalGNN on ALL metrics. Vision integration (DINOv2) adds complexity without benefit on ALOHA — spatial predicates are solvable from joint positions alone.

## Overview

This project implements an **MCP-to-ROS 2 Bridge** that allows any AI model (Claude, Llama, GPT, etc.) to control robots through a standardized protocol. The key innovation is treating the AI "brain" as a swappable component—similar to how USB-C standardizes hardware connections.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLOUD / WORKSTATION                          │
│  ┌───────────────┐    ┌───────────────────┐    ┌─────────────────┐  │
│  │   LLM Agent   │◄──►│   MCP Client SDK  │◄──►│  GNN Reasoner   │  │
│  │ (Llama/Qwen)  │    │   (JSON-RPC/SSE)  │    │ (World Graph)   │  │
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
- **🔄 Swappable AI**: Change the "brain" (Llama ↔ Qwen) without modifying robot code
- **🧠 Semantic Perception**: GNN-based world graph for structured environment understanding
- **🤖 LeRobot Integration**: Train on HuggingFace LeRobot datasets (ALOHA, PushT, etc.)
- **📊 Explainability**: All AI decisions logged as tool calls and resource queries
- **☁️ Cloud-Edge Split**: Heavy compute on cloud, lightweight execution on robot
- **📈 Predicate Prediction**: 9 spatial/interaction predicates with 97% accuracy
- **🔮 Pre-Execution Simulation**: ForwardDynamicsModel verifies LLM plans before execution
- **⏱️ Temporal Stability**: SpatiotemporalGNN with GRU eliminates predicate flicker

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

#### Option A: LeRobot GNN Pipeline (No ROS required)

```bash
source .venv/bin/activate

# Run with synthetic data (quick test)
python scripts/demo_lerobot_pipeline.py --synthetic --frames 100

# Run with real LeRobot dataset
python scripts/demo_lerobot_pipeline.py --repo lerobot/aloha_static_coffee --frames 200

# Train the GNN model
python scripts/train_relational_gnn.py --epochs 100 --dataset aloha
```

#### Option B: Full ROS 2 + Gazebo Demo

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

# With Llama (requires Ollama running locally)
python scripts/run_experiment.py --agent llama --goal "Navigate to position (3, -3)"

# With Qwen (requires Ollama running locally)
python scripts/run_experiment.py --agent qwen --goal "Navigate to position (3, -3)"
```

#### Option C: LeRobot + Ollama Demo (No ROS required)

This is the simplest way to test the full LLM → MCP → GNN pipeline:

**Step 1: Install Ollama**
```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull llama3.2 (3B, 2.0GB) and/or qwen2.5 (3B, 1.9GB)
ollama pull llama3.2
ollama pull qwen2.5:3b

# Verify
ollama list
```

**Step 2: Start MCP Server with LeRobot**
```bash
source .venv/bin/activate
python -m mcp_ros2_bridge.server --lerobot
```

**Step 3: Run LLM Agent (new terminal)**
```bash
source .venv/bin/activate

# Llama (slightly slower, uses more steps)
python scripts/run_experiment.py --agent llama --goal "Query world graph and report predicates"

# Qwen (40% faster, fewer steps) ✅ RECOMMENDED
python scripts/run_experiment.py --agent qwen --goal "Query world graph and report predicates"
```

**Expected Output:**
```json
{
  "success": true,
  "total_steps": 2,
  "duration_seconds": 1.5,
  "observation": {
    "world_graph": {"num_nodes": 16, "num_edges": 54},
    "spatial_predicates": 78,
    "interaction_predicates": 42
  }
}
```

## Project Structure

```
AI2MCP/
├── src/
│   ├── mcp_ros2_bridge/      # MCP server with ROS 2 integration
│   │   ├── server.py         # SSE-based MCP server
│   │   ├── ros_node.py       # ROS 2 node wrapper
│   │   ├── tools/            # MCP tools (motion, perception, prediction)
│   │   │   ├── motion.py     # Movement commands
│   │   │   ├── perception.py # Sensor queries
│   │   │   └── prediction.py # GNN-based prediction tools
│   │   └── resources/        # MCP resources (pose, scan, world_graph)
│   │       ├── lerobot_state.py  # LeRobot dataset resources
│   │       └── world_graph.py    # Semantic graph resources
│   │
│   ├── gnn_reasoner/         # Semantic scene understanding
│   │   ├── data_manager.py   # LeRobot dataset loading
│   │   ├── lerobot_transformer.py  # State → Graph conversion
│   │   ├── benchmark.py      # Performance metrics (pass@k, latency)
│   │   ├── detector.py       # DETIC/GroundingDINO object detection
│   │   ├── depth.py          # ZoeDepth/MiDaS depth estimation
│   │   ├── camera.py         # Camera intrinsics & 3D projection
│   │   ├── graph_builder.py  # Sensor → World Graph
│   │   └── model/            # PyTorch Geometric GNN
│   │       ├── relational_gnn.py   # Kinematic GNN (97.03% acc) ✅ RECOMMENDED
│   │       ├── multimodal_gnn.py   # Vision+Kinematic GNN (96.51% acc)
│   │       ├── forward_dynamics.py # Pre-execution simulation (259K params)
│   │       ├── spatiotemporal_gnn.py # Temporal stability (~90% acc)
│   │       └── scene_gnn.py        # Scene understanding
│   │
│   └── agents/               # Swappable AI agents
│       ├── base_agent.py     # Abstract agent interface + MCPClient
│       ├── llama_agent.py    # Llama3.2 via Ollama
│       └── qwen_agent.py     # Qwen2.5 via Ollama ✅ RECOMMENDED
│
├── experiments/              # Training & benchmark results
│   ├── aloha_training/       # Local kinematic GNN (99.4% acc, 5k frames)
│   ├── remote_training/      # Full 55k frame training (RTX 3070)
│   │   ├── relational_gnn/   # 97.03% acc ✅ BEST
│   │   ├── multimodal_gnn_55k_v2/  # 96.51% acc
│   │   ├── forward_dynamics_e2e/   # δ=0.0017, conf=0.49-0.62
│   │   └── spatiotemporal_gnn/     # ~90% acc (temporal)
│   ├── comparison_final_real/  # Fair A vs C comparison
│   ├── ablation_depth/       # Depth noise ablation
│   ├── agent_benchmark.json  # Llama vs Qwen results
│   └── training/             # Synthetic baseline (95.9% acc)
│
├── figures/                  # Thesis figures (28 PNGs, auto-generated)
│   ├── training_curves.png   # Loss/accuracy plots
│   ├── architecture.png      # System diagram
│   ├── comparison/           # A vs C comparison figures
│   ├── forward_dynamics_*.png  # Phase 10 figures
│   └── stgnn_*.png           # Phase 11 figures
│
├── simulation/               # Gazebo simulation setup
│   ├── launch/              # ROS 2 launch files
│   ├── config/              # Nav2 parameters
│   └── worlds/              # Custom Gazebo worlds
│
├── scripts/                 # Utility scripts
│   ├── train_relational_gnn.py   # Kinematic GNN training
│   ├── train_multimodal_gnn.py   # MultiModal GNN training
│   ├── train_forward_model.py    # ForwardDynamicsModel training
│   ├── train_spatiotemporal_gnn.py # SpatiotemporalGNN training
│   ├── compare_models.py         # A vs C benchmark
│   ├── benchmark_agents.py       # Llama vs Qwen agent benchmark
│   ├── demo_lerobot_pipeline.py  # Pipeline demo
│   ├── run_experiment.py         # LLM agent runner
│   ├── generate_thesis_figures.py
│   └── generate_comparison_figures.py
│
├── tests/                   # Test suite
└── docs/                    # Documentation
```

## MCP Interface

### Tools (Actions)

#### Motion Tools
| Tool | Description |
|------|-------------|
| `move(linear_x, angular_z, duration_ms)` | Move with velocity for duration |
| `stop()` | Emergency stop |
| `rotate(angle_degrees)` | Rotate in place |
| `move_forward(distance_meters)` | Move forward by distance |

#### Perception Tools
| Tool | Description |
|------|-------------|
| `get_obstacle_distances(directions)` | Query obstacles in directions |
| `check_path_clear(distance, width)` | Check if path is navigable |
| `scan_surroundings(num_sectors)` | 360° obstacle scan |

#### Prediction Tools (LeRobot/GNN)
| Tool | Description |
|------|-------------|
| `get_world_graph(frame_idx)` | Get semantic graph with predicates |
| `predict_action_outcome(action, num_steps)` | Predict future state changes |
| `advance_frame()` | Move to next frame in trajectory |
| `set_frame(index)` | Jump to specific frame |
| `get_predicates(threshold)` | Get active spatial/interaction predicates |
| `simulate_action(action_sequence, confidence_threshold)` | Pre-execution verification |
| `project_future(action, horizon_steps)` | **NEW** Temporal predicate projection |

#### Pre-Execution Simulation (Phase 10) ✅

The `simulate_action` tool enables LLM agents to **verify plans before physical execution**:

```python
# LLM proposes action sequence
result = await client.call_tool("simulate_action", {
    "action_sequence": [[0.1, 0.2, ...], [0.15, 0.25, ...]],  # 14-DoF actions
    "num_steps": 5,
    "confidence_threshold": 0.7
})

# Returns: {"recommendation": "EXECUTE" | "REPLAN", "confidence": 0.85, ...}
```

| Metric | Value |
|--------|-------|
| Model | ForwardDynamicsModel (259K params) |
| Training | 55k frames, 17 min pre-compute + 2.3 min training |
| Inference | 41ms |
| Delta Error | 0.0017 (1.7mm accuracy) |
| Confidence Range | 0.49–0.62 |

#### Temporal Stability (Phase 11) ✅

The `project_future` tool uses SpatiotemporalGNN to **predict future predicates**:

```python
# AI asks: "If I move forward, what predicates will be active?"
result = await client.call_tool("project_future", {
    "action": [0.1, 0.0, ...],  # 14-DoF action
    "horizon_steps": 3
})

# Returns: predicted predicates at t+1, t+2, t+3 with confidence scores
```

| Metric | Value |
|--------|-------|
| Model | SpatiotemporalGNN (GRU + RelationalGNN) |
| Training | 55k frames, 47 min |
| Accuracy | ~90% |
| Sequence Length | 5 frames |

### Available LLM Agents

| Agent | Model | Backend | Status |
|-------|-------|---------|--------|
| **QwenAgent** | qwen2.5:3b (1.9GB) | Ollama | ✅ **RECOMMENDED** (40% faster) |
| LlamaAgent | llama3.2 (2.0GB) | Ollama | ✅ Validated |

Both agents achieve **100% success rate** on standardized goals. Qwen uses fewer steps and is faster overall.

### Resources (State)

#### Robot Resources
| Resource URI | Description |
|--------------|-------------|
| `robot://pose` | Current position (x, y, θ) |
| `robot://velocity` | Current velocity |
| `robot://scan/summary` | LiDAR scan summary by quadrant |
| `robot://scan/obstacles` | Detected obstacle clusters |
| `robot://world_graph` | Semantic scene graph |

#### LeRobot Resources
| Resource URI | Description |
|--------------|-------------|
| `lerobot://current_state` | Current observation state (14 joints) |
| `lerobot://world_graph` | GNN-processed relational graph |
| `lerobot://predicates` | Active predicates with confidence scores |
| `lerobot://dataset_info` | Dataset metadata (episodes, frames) |

### Predicate Types

The GNN predicts 9 binary predicates:

**Spatial (5):** `is_near`, `is_above`, `is_below`, `is_left_of`, `is_right_of`

**Interaction (4):** `is_holding`, `is_contacting`, `is_approaching`, `is_retracting`

## Thesis Validation

The key experiment demonstrates **swappable intelligence**:

1. Run task with Llama3.2 agent
2. Run identical task with Qwen2.5 agent  
3. **No robot-side code changes** between runs

```bash
# Run agent benchmark (10 goals, both agents)
python scripts/benchmark_agents.py

# Or run single agent
python scripts/run_experiment.py --agent llama --goal "Get world graph"
python scripts/run_experiment.py --agent qwen --goal "Get world graph"
```

Results are saved to `experiments/agent_benchmark.json`.

## Experiment Results

### GNN Training Performance

| Model | Dataset | Frames | Accuracy | Training Time | GPU |
|-------|---------|--------|----------|---------------|-----|
| Synthetic Baseline | Synthetic | 1k | 95.9% | 21s | RTX 500 |
| RelationalGNN (local) | ALOHA | 5k | 99.4% | 205s | RTX 500 |
| **RelationalGNN** | ALOHA | **55k** | **97.03%** | 29 min | RTX 3070 |
| MultiModalGNN | ALOHA | 55k | 96.51% | 31 min | RTX 3070 |
| ForwardDynamicsModel | ALOHA | 55k | δ=0.0017 | 2.3 min | RTX 3070 |
| **SpatiotemporalGNN** | ALOHA | 55k | **~90%** | 47 min | RTX 3070 |

> **Note**: The 5k-frame local training shows higher accuracy (99.4%) than 55k remote (97.03%) due to overfitting on the smaller dataset. The 55k results are more representative.

### Inference Benchmark (200 frames, trained model)

| Metric | Value |
|--------|-------|
| Predicate Accuracy | 92.5% |
| Precision | 84.5% |
| Recall | 79.9% |
| F1 Score | 82.1% |
| Pass@1 | 88.2% |
| Pass@3 | 98.2% |
| Inference Latency (mean) | 12.6ms |
| Inference Latency (p95) | 12.7ms |
| Graph Construction | 1.7ms |
| Serialization | 0.15ms |
| Protocol Overhead | 30.4% |

### Predicate Distribution (200 frames)

| Predicate | Detections |
|-----------|------------|
| `is_near` | 8,539 |
| `is_right_of` | 4,769 |
| `is_left_of` | 4,727 |
| `is_above` | 0 |
| `is_below` | 0 |

*Interaction predicates (`is_holding`, `is_approaching`, etc.) require intentional manipulation behavior not present in synthetic test data.*

### Reproducibility

```bash
# Reproduce training
python scripts/train_relational_gnn.py --epochs 100 --dataset aloha

# Reproduce benchmark
python scripts/demo_lerobot_pipeline.py --synthetic --frames 200 \
    --model experiments/aloha_training/best_model.pt \
    --output experiments/benchmark_reproduced.json

# Generate thesis figures
python scripts/generate_thesis_figures.py --output figures/
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SERVER_HOST` | `0.0.0.0` | MCP server bind address |
| `MCP_SERVER_PORT` | `8080` | MCP server port |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Ollama model name (qwen2.5:3b or llama3.2) |

### Agent Configuration

```python
from agents.base_agent import AgentConfig
from agents.qwen_agent import QwenAgent  # or LlamaAgent

config = AgentConfig(
    mcp_server_url="http://localhost:8080",
    max_steps=10,
    timeout_seconds=30.0,
)

agent = QwenAgent(config=config, model="qwen2.5:3b")  # ✅ RECOMMENDED
# or: agent = LlamaAgent(config=config, model="llama3.2")
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

## Known Issues & Limitations

| Issue | Status | Workaround |
|-------|--------|------------|
| MCP SSE resource transport bug | ⚠️ Library bug | Agent uses tool results instead of resources |
| LLM sends string numbers (`"0"` vs `0`) | ✅ Fixed | Auto-coerced in `MCPClient.call_tool()` |
| Llama 3B loops on complex prompts | ✅ Fixed | Simplified system prompt with explicit rules |
| ZoeDepth installation (timm version) | ⚠️ | Falls back to MiDaS (relative depth only) |
| `is_holding`/`is_contacting` = 0.000 F1 | ⚠️ Data limitation | ALOHA lacks contact annotations |

### Contact Predicates Limitation

`is_holding` and `is_contacting` show 0.000 F1 on real ALOHA data because:
- Only ~0.7% positive `is_holding` edges in dataset
- No explicit contact annotations available
- Heuristic labels (gripper near object + closed) are insufficient

**Solution**: Requires annotated dataset with explicit contact labels, F/T sensing, or tactile integration.

> **Note on Resource Bug**: MCP SDK v1.25.0 has a bug in the SSE transport layer for resources. Tools work correctly. The agent uses `get_world_graph` tool calls instead of resource reads.

## Vision Integration

This project includes two approaches for integrating visual object detection with the kinematic GNN:

### Option A: RelationalGNN (Kinematic + Geometric Fusion) ✅ RECOMMENDED
```
JointState → Graph → RelationalGNN → Predicates
(Optional) Image → GroundingDINO → Depth → 3D Objects → Graph
```
- **Latency:** 1.5ms (GNN only) / 297ms (with real vision)
- **Accuracy:** 97.03%
- **Best for:** All use cases on ALOHA-style datasets

### Option C: MultiModalGNN (DINOv2 Cross-Attention)
```
Image → DINOv2 → RoI Pool → Cross-Attention → MultiModalGNN
```
- **Latency:** 24ms
- **Accuracy:** 96.51%
- **Best for:** Datasets where objects are NOT encoded in kinematics

### Fair Comparison Results (55k vs 55k frames)

| Metric | Option A | Option C | Winner |
|--------|----------|----------|--------|
| Micro Accuracy | **97.03%** | 96.51% | **A (+0.5%)** |
| Macro F1 | **0.358** | 0.348 | **A (+2.9%)** |
| `is_near` F1 | **0.954** | 0.920 | **A** |
| `is_approaching` F1 | **0.182** | 0.156 | **A** |
| Latency | **1.5ms** | 24ms | **A (16× faster)** |
| Peak Memory | **19.4MB** | 141.8MB | **A (7× less)** |
| Model Size | **0.81MB** | 2.14MB | **A (2.6× smaller)** |

### Honest E2E Latency (Real Vision on RTX 3070)

| Component | Time |
|-----------|------|
| GroundingDINO detection | 217-234ms |
| Depth Anything V2 | 61ms |
| GNN inference | 1.4ms |
| **Total E2E** | **297-332ms** |

> ⚠️ **Note**: The 2.4ms latency previously reported was with **mock detectors**. Real vision adds ~300ms.

```bash
# Train MultiModalGNN
python scripts/train_multimodal_gnn.py --epochs 100 --output experiments/multimodal

# Compare models
python scripts/compare_models.py --model-a experiments/aloha_training/best_model.pt \
    --model-c experiments/multimodal_aloha/best_model.pt --frames 500

# Run ablation study
python scripts/ablation_depth_noise.py --frames 200 --output experiments/ablation_depth
```

### Hardware Requirements

| Mode | GPU | VRAM | Stack |
|------|-----|------|-------|
| **Development** | Any (CPU OK) | - | Mock detectors |
| **Kinematic GNN** | RTX 500+ | 1GB | RelationalGNN only |
| **MultiModal (Heavy)** | RTX 4080+ | 4GB+ | GroundingDINO + ZoeDepth + DINOv2 |
| **MultiModal (Edge)** | RTX 500 | 1GB | YOLO-World + DepthAnything V2 Small |

> **Note**: The "heavy" vision stack (GroundingDINO + ZoeDepth + DINOv2) requires ~3.8GB VRAM. For RTX 500 (4GB) deployment, use mock detectors during development or the edge-native stack (YOLO-World + Depth Anything V2 Small, ~0.9GB).

---

## Research Contributions

1. **N×M → N+M Complexity**: Single MCP interface per robot connects to any model
2. **Explainable Robot AI**: All decisions logged as structured tool calls
3. **Semantic Perception**: GNN-processed world graphs with 97% predicate accuracy
4. **Protocol-Driven Robotics**: Foundation for multi-robot, multi-agent systems
5. **LeRobot Integration**: First MCP bridge for HuggingFace robotics datasets
6. **Swappable AI Validated**: Llama3.2 + Qwen2.5 → MCP → GNN E2E (100% success rate)
7. **Pre-Execution Simulation**: ForwardDynamicsModel for LLM plan verification before physical execution
8. **Fair Architecture Comparison**: RelationalGNN vs MultiModalGNN on 55k frames — kinematic wins
9. **Temporal Predicate Stability**: SpatiotemporalGNN with GRU eliminates frame-to-frame flicker (~90% acc)
10. **Agent Benchmark**: Qwen 40% faster than Llama, 65% fewer steps to complete tasks

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
- [LeRobot](https://huggingface.co/lerobot) by HuggingFace for robotics datasets
- [ALOHA](https://tonyzhaozh.github.io/aloha/) for bimanual manipulation data

