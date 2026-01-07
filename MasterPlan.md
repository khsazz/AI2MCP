# AI2MCP Master Plan

> **Purpose**: Strategic planning and progress tracking for the thesis project.  
> **Detailed Logging**: See `CONTEXT_DUMP.txt` for implementation details, code patterns, and problem solutions.  
> **Last Updated**: 2026-01-06

---

## Project Overview

**Title**: A Standardized Middleware Architecture for Decoupled Robotic Intelligence using the Model Context Protocol (MCP)

**Core Thesis Claim**: Robotic intelligence can be treated as a swappable service—change the AI "brain" (Claude ↔ Llama) without modifying robot code.

**Key Results Achieved**:
| Metric | Value | Status |
|--------|-------|--------|
| Kinematic GNN Accuracy | 99.4% | ✅ |
| MultiModal GNN Accuracy | 96.2% | ✅ |
| `is_near` F1 Improvement | +35.6% (vision fusion) | ✅ |
| `is_holding` F1 | 0.914 (was 0.000) | ✅ |
| `is_contacting` F1 | 0.960 (was 0.000) | ✅ |
| Macro F1 | 0.671 (was 0.336) | ✅ |
| Llama Agent E2E | 3 steps, 5.3s | ✅ |
| Swappable AI Validated | LLM → MCP → GNN | ✅ |

---

## Phase Summary

| Phase | Focus | Status | Duration |
|-------|-------|--------|----------|
| 1 | Core Infrastructure | ✅ Complete | Pre-2025-01-05 |
| 2 | Vision Integration | ✅ Complete | 2025-01-05 |
| 3 | Vision Finalization | ✅ Complete | 2025-01-06 |
| 4 | LLM Agent Integration | ✅ Complete | 2025-01-06 |
| 5 | Ollama/Llama3.2 E2E | ✅ Complete | 2026-01-06 |
| 6 | Thesis Documentation | ✅ Complete | Ongoing |
| 7 | Edge Deployment | ⏸️ Deferred | — |
| 8 | Physical Robot | ❌ Out of Scope | — |

---

## Phase 1: Core Infrastructure ✅

**Objective**: Build MCP-ROS2 bridge with GNN-based semantic understanding.

| Task | Status | Notes |
|------|--------|-------|
| RelationalGNN architecture | ✅ | 203K params, GATv2, 3 layers |
| Training pipeline | ✅ | GPU profiles (RTX 500/4080/CPU) |
| MCP server (SSE transport) | ✅ | 13 tools, 13 resources |
| Tool registration refactor | ✅ | Consolidated handler pattern |
| LeRobot integration | ✅ | DataManager, GraphTransformer |
| Synthetic baseline | ✅ | 95.9% accuracy, 21s training |
| ALOHA training | ✅ | **99.4% accuracy**, 205s |

**Key Files**:
- `src/mcp_ros2_bridge/server.py` — MCP server
- `src/gnn_reasoner/model/relational_gnn.py` — GNN model
- `src/gnn_reasoner/lerobot_transformer.py` — State → Graph
- `experiments/aloha_training/best_model.pt` — Trained checkpoint

---

## Phase 2: Vision Integration ✅

**Objective**: Add visual object detection to kinematic-only GNN.

| Task | Status | Notes |
|------|--------|-------|
| `detector.py` (DETIC/GroundingDINO/YOLOv8) | ✅ | Mock + real implementations |
| `depth.py` (ZoeDepth/MiDaS/DepthAnything) | ✅ | Mock + real implementations |
| `camera.py` (intrinsics, 3D projection) | ✅ | `CameraIntrinsics.default_aloha()` |
| Extended `LeRobotGraphTransformer` | ✅ | `to_graph_with_objects()` |
| Option A: Geometric Fusion | ✅ | 2.4ms latency, 92.3% acc |
| Option C: MultiModal Fusion | ✅ | 52ms latency, 96.2% acc |
| Unit tests (40 total) | ✅ | `test_vision_pipeline.py`, `test_multimodal_gnn.py` |

**Key Results**:
| Metric | Option A | Option C |
|--------|----------|----------|
| Micro Accuracy | 92.3% | **96.2%** |
| `is_near` F1 | 0.67 | **0.91** (+35%) |
| Latency | **2.4ms** | 52ms |
| Memory | **107MB** | 231MB |

**Decision**: Use-case driven selection
- Option A → Real-time control
- Option C → Planning/reasoning

**Key Files**:
- `src/gnn_reasoner/detector.py`
- `src/gnn_reasoner/depth.py`
- `src/gnn_reasoner/camera.py`
- `src/gnn_reasoner/model/multimodal_gnn.py`
- `experiments/multimodal_aloha/best_model.pt`

---

## Phase 3: Vision Finalization ✅

**Objective**: Resolve open questions, document limitations.

| Task | Status | Resolution |
|------|--------|------------|
| Camera intrinsics source | ✅ | `CameraIntrinsics.default_aloha()` (60° FOV) |
| Detector choice | ✅ | GroundingDINO (dev), YOLO-World (edge) |
| VRAM constraints documented | ✅ | Heavy stack ~3.8GB, RTX 500 limit 4GB |
| README hardware section | ✅ | Added requirements table |

**Hardware Recommendations**:
| Mode | GPU | VRAM | Stack |
|------|-----|------|-------|
| Development | Any | — | Mock detectors |
| Kinematic GNN | RTX 500+ | 1GB | RelationalGNN only |
| MultiModal (Heavy) | RTX 4080+ | 4GB+ | GroundingDINO + ZoeDepth + DINOv2 |
| MultiModal (Edge) | RTX 500 | 1GB | YOLO-World + DepthAnything V2 Small |

---

## Phase 4: LLM Agent Integration ✅

**Objective**: Connect LLM agents to MCP server with GNN context.

| Task | Status | Notes |
|------|--------|-------|
| MCPClient → real MCP SDK (SSE) | ✅ | HTTP fallback included |
| Tool/resource caching | ✅ | On connect |
| System prompts (Claude/Llama) | ✅ | Includes prediction tools, predicates |
| Conversation history | ✅ | Last 5 turns |
| `observe()` with GNN predicates | ✅ | Extracts from tool results |
| Claude agent testing | ⏸️ | Deferred (Llama prioritized) |

**Agent Capabilities**:
- **Motion**: `move`, `stop`, `rotate`, `move_forward`
- **Perception**: `get_obstacle_distances`, `check_path_clear`, `scan_surroundings`
- **Prediction (GNN)**: `get_world_graph`, `get_predicates`, `advance_frame`, `set_frame`

**Key Files**:
- `src/agents/base_agent.py` — MCPClient, AgentState
- `src/agents/claude_agent.py`
- `src/agents/llama_agent.py`

---

## Phase 5: Ollama + Llama3.2 E2E ✅

**Objective**: Validate swappable AI thesis with local LLM.

| Task | Status | Notes |
|------|--------|-------|
| Ollama installation | ✅ | NVIDIA GPU detected |
| llama3.2 model pull | ✅ | 3B params, 2.0GB |
| Argument type coercion fix | ✅ | String → number auto-convert |
| Simplified observe() | ✅ | Uses tool results + resources |
| E2E test | ✅ | **3 steps, 5.3s, 16 nodes, 54 edges** |
| MCP resource bug fix | ✅ | Route→Mount, ReadResourceContents, str(uri) |
| Zero-Holding fix | ✅ | Global Feature Conditioning + Data Balancing |

**Problems Solved** (see CONTEXT_DUMP.txt for details):
1. Llama infinite loop → Simplified system prompt with step limits
2. String numbers (`"0"` vs `0`) → `_coerce_argument_types()`
3. MCP resource SSE bug → Fixed: Mount not Route, ReadResourceContents type, str(uri)
4. Zero-Holding anomaly → ConditionalPredicateHead with gripper state conditioning

---

## Phase 6: Thesis Documentation ✅

**Objective**: Generate figures, write thesis chapters.

| Task | Status | Notes |
|------|--------|-------|
| Training curves figure | ✅ | `figures/training_curves.pdf` |
| Architecture diagram | ✅ | `figures/architecture.pdf` |
| Pass@k figure | ✅ | `figures/pass_at_k.pdf` |
| Classification metrics | ✅ | `figures/classification_metrics.pdf` |
| Comparison figures (6) | ✅ | `thesis/figures/` |
| Ablation figures (3) | ✅ | `experiments/ablation_depth/` |
| LaTeX thesis (7 chapters) | ✅ | `thesis/main.tex` |
| BibTeX references | ✅ | 25+ entries |

**Thesis Chapters**:
1. Introduction — motivation, research question, contributions
2. Background — MCP, ROS2, GNN, LeRobot, related work
3. Methodology — architecture, tools, resources, GNN design
4. Implementation — code structure, training pipeline
5. Results — training curves, predicates, latency, throughput
6. Discussion — implications, limitations, future work
7. Conclusion — summary, contributions

**Key Thesis Claims**:
- MCP solves N×M integration problem
- GNN provides structured relational understanding (99.4% accuracy)
- Vision integration improves `is_near` F1 by +35.6%
- Swappable AI brain validated (Llama3.2 → MCP → GNN E2E)

---

## Phase 7: Edge Deployment ⏸️ DEFERRED

**Objective**: Deploy on RTX 500 with lightweight vision stack.

| Task | Status | Blocker |
|------|--------|---------|
| Add YOLO-World to `detector.py` | ⬜ | Requires integration work |
| Add Depth Anything V2 Small to `depth.py` | ⬜ | Requires integration work |
| Create `--profile edge` flag | ⬜ | Depends on above |
| Document edge deployment path | ⬜ | — |
| Validate <1GB VRAM usage | ⬜ | — |

**Target Stack** (~0.9GB):
- YOLO-World-S (0.5GB)
- Depth Anything V2 Small (0.3GB)
- RelationalGNN (0.1GB)

**Priority**: Low (thesis core validated, edge is future work)

---

## Phase 8: Physical Robot ❌ OUT OF SCOPE

**Status**: Not pursuing for this thesis.

**Rationale**: 
- Core thesis claim (swappable AI) validated via simulation + LeRobot datasets
- Hardware deployment is engineering work, not research contribution
- Safety constraint verification requires dedicated study

| Task | Status | Notes |
|------|--------|-------|
| Hardware access | ❌ | Out of scope |
| ROS 2 integration testing | ❌ | Out of scope |
| Gazebo simulation validation | ⏸️ | Optional, low priority |
| Real-world latency benchmarks | ❌ | Out of scope |
| Safety constraint verification | ❌ | Out of scope |

---

## Future Research Directions

| Direction | Priority | Notes |
|-----------|----------|-------|
| Claude agent testing | Medium | Deferred, revisit after thesis submission |
| Temporal GNN for motion predicates | Low | `is_approaching`/`is_retracting` still 0 (need velocity) |
| LLM chain-of-thought reasoning | Low | Enhanced planning capabilities |
| Multi-robot orchestration via MCP | Low | Scalability demonstration |
| Benchmark suite standardization | Low | Reproducibility for community |
| Additional datasets (RLBench, BridgeData) | Low | Generalization study |

### Explicitly Out of Scope

| Item | Reason |
|------|--------|
| Physical robot deployment | Engineering work, not research contribution |
| Safety constraint verification | Requires dedicated safety study |

---

## Quick Reference

### CLI Commands

```bash
# Start MCP server with LeRobot
python -m mcp_ros2_bridge.server --lerobot

# Train kinematic GNN
python scripts/train_relational_gnn.py --epochs 100 --dataset aloha

# Train multimodal GNN
python scripts/train_multimodal_gnn.py --epochs 100 --output experiments/multimodal

# Run comparison benchmark
python scripts/compare_models.py --model-a experiments/aloha_training/best_model.pt \
    --model-c experiments/multimodal_aloha/best_model.pt --frames 500

# Run Llama agent
python scripts/run_experiment.py --agent llama --goal "Query world graph"

# Generate thesis figures
python scripts/generate_thesis_figures.py
python scripts/generate_comparison_figures.py
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SERVER_PORT` | `8080` | MCP server port |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `LEROBOT_REPO_ID` | `lerobot/aloha_static_coffee` | Dataset |

### Key Checkpoints

| Checkpoint | Location | Accuracy |
|------------|----------|----------|
| Kinematic GNN (best) | `experiments/aloha_training/best_model.pt` | 99.4% |
| MultiModal GNN (best) | `experiments/multimodal_aloha/best_model.pt` | 96.2% |
| Synthetic baseline | `experiments/training/best_model.pt` | 95.9% |

---

## Document Conventions

### Status Markers
- ✅ **Done** — Task complete
- ⏸️ **Deferred** — Planned but postponed
- ⬜ **Pending** — Not started
- ⚠️ **Blocked** — Cannot proceed (see blocker)
- ❌ **Out of Scope** — Explicitly excluded from thesis

### Cross-References
- **Detailed logs**: `CONTEXT_DUMP.txt`
- **Vision details**: `VISION_INTEGRATION_PLAN.md`
- **User docs**: `README.md`

### Update Protocol
1. Update this file when phase status changes
2. Add detailed implementation notes to `CONTEXT_DUMP.txt`
3. Keep README.md as user-facing documentation

---

*Generated from project analysis. See CONTEXT_DUMP.txt for implementation history.*

