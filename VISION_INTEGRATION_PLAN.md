# Vision Integration Plan

## Status: Planning Phase
Last Updated: 2025-01-05

---

## Decision Log

| Decision | Choice | Rationale | Date |
|----------|--------|-----------|------|
| Option B (z=0 heuristic) | ❌ Rejected | Inaccurate for stacked objects | 2025-01-05 |
| Implementation scope | Both A + C | Comparative study for thesis | 2025-01-05 |
| Vision encoder | DINOv2-B | Higher accuracy, acceptable latency | 2025-01-05 |
| Depth model | ZoeDepth | State-of-art monocular depth | 2025-01-05 |
| Object detector | DETIC | Open-vocabulary capability | 2025-01-05 |

---

## Architecture Overview

### Option A: Geometric Fusion (Service-Based)
```
Image → DETIC → Bboxes
                  ↓
Image → ZoeDepth → Depth Map
                  ↓
        Camera Projection → 3D Object Positions
                  ↓
        Merge with Kinematic Graph (type=2 nodes)
                  ↓
        Existing RelationalGNN
```

### Option C: Learned Fusion (Multi-Modal GNN)
```
Image → DINOv2-B → Patch Embeddings
                      ↓
        RoI Pooling over DETIC boxes
                      ↓
        Vision Node Features (d=768)
                      ↓
        Cross-Attention with Kinematic Nodes
                      ↓
        Unified GNN Layers
                      ↓
        PredicateHead
```

---

## Implementation Phases

### Phase 1: Shared Infrastructure [NOT STARTED]
- [ ] `src/gnn_reasoner/detector.py` — DETIC wrapper
- [ ] `src/gnn_reasoner/depth.py` — ZoeDepth wrapper  
- [ ] `src/gnn_reasoner/camera.py` — Projection utilities
- [ ] `src/gnn_reasoner/data_manager.py` — Add `get_image()` method
- [ ] Unit tests for each component

### Phase 2: Option A Integration [NOT STARTED]
- [ ] Extend `LeRobotGraphTransformer.to_graph()` to accept detections
- [ ] Add object node creation logic
- [ ] Update `compute_heuristic_predicates()` for object nodes
- [ ] Validate with existing `RelationalGNN` weights
- [ ] Benchmark latency

### Phase 3: Option C Implementation [NOT STARTED]
- [ ] `src/gnn_reasoner/model/multimodal_gnn.py`
- [ ] Vision encoder loading (frozen DINOv2-B)
- [ ] Cross-attention fusion layer
- [ ] Training script modifications
- [ ] Checkpoint management

### Phase 4: Evaluation [NOT STARTED]
- [ ] Manual annotation of 200 test frames
- [ ] Benchmark harness for A vs C comparison
- [ ] Statistical significance tests
- [ ] Generate thesis figures

---

## Dataset Strategy

### Training Scope: MODERATE
Total target: ~80k frames across 2 datasets

### Primary: LeRobot ALOHA (lerobot/aloha_static_coffee)
- **Frames:** 55,000 (full dataset)
- **Tasks:** Coffee making (single task)
- **Objects:** Cup, coffee machine, table
- **Depth:** Estimated via ZoeDepth
- **Status:** Already downloaded
- **Used for:** Primary training, real-world evaluation

### Secondary: RLBench (Simulation)
- **Frames:** ~25,000 (5 tasks × 100 episodes × 50 frames)
- **Tasks:** 
  1. `pick_and_lift` — basic grasping
  2. `put_item_in_drawer` — container interaction
  3. `stack_blocks` — multi-object reasoning
  4. `place_cups` — transfer to ALOHA domain
  5. `push_button` — contact detection
- **Objects:** Diverse (cubes, cylinders, cups, drawers, buttons)
- **Depth:** Perfect (from simulation)
- **Object poses:** Ground truth (from simulation)
- **Used for:** 
  - Ablation: estimated vs GT depth
  - Ablation: object diversity
  - Generalization test (train ALOHA → test RLBench)

### NOT Using (Out of Scope)
- ❌ BridgeData V2 — too large, redundant with RLBench
- ❌ DROID — requires extensive preprocessing
- ❌ Open X-Embodiment — heterogeneous robots

---

## Ground Truth Strategy

### ALOHA Dataset Splits
| Split | Labeling Method | Size | Notes |
|-------|-----------------|------|-------|
| Train | Heuristic derivation | 45k frames | Automated |
| Val | Heuristic derivation | 5k frames | Automated |
| Test | Manual annotation | 200 frames | Human labeled |

### RLBench Dataset Splits  
| Split | Labeling Method | Size | Notes |
|-------|-----------------|------|-------|
| Train | Simulation ground truth | 20k frames | Perfect labels |
| Val | Simulation ground truth | 2.5k frames | Perfect labels |
| Test | Simulation ground truth | 2.5k frames | Perfect labels |

### Heuristic Label Computation (ALOHA)
```python
# Spatial predicates (kinematic only)
is_near = distance < 0.2m
is_above = delta_z > 0.1m
is_left_of = delta_x < -0.05m

# Interaction predicates (require object detection)
is_holding = (gripper_open < 0.3) AND (object_distance < 0.05m)
is_contacting = object_distance < 0.08m
is_approaching = dot(velocity, object_direction) > 0
```

### Manual Annotation Protocol (200 ALOHA test frames)
1. Sample 200 frames uniformly across episodes
2. For each frame, annotator labels:
   - Which objects are visible (bounding boxes)
   - For each gripper-object pair: is_holding, is_contacting
3. Tools: Label Studio or CVAT
4. Estimated time: ~4 hours

---

## Model Configurations

### DINOv2 Variants
| Variant | Params | Feature Dim | Latency (RTX 500) | Status |
|---------|--------|-------------|-------------------|--------|
| DINOv2-S | 22M | 384 | ~15ms | Backup |
| DINOv2-B | 86M | 768 | ~25ms | **Selected** |
| DINOv2-L | 300M | 1024 | ~60ms | Too slow |

### DETIC Configuration
- Backbone: Swin-B
- Vocabulary: LVIS (1203 classes) + custom prompts
- Expected latency: ~50ms/frame

### ZoeDepth Configuration
- Model: ZoeD_NK (NYU+KITTI pretrained)
- Output: Metric depth in meters
- Expected latency: ~30ms/frame

---

## Expected Latency Breakdown

### Option A (Geometric Fusion)
| Component | Latency |
|-----------|---------|
| DETIC detection | 50ms |
| ZoeDepth | 30ms |
| Camera projection | 1ms |
| Graph construction | 2ms |
| RelationalGNN | 15ms |
| **Total** | **~98ms** |

### Option C (Learned Fusion)
| Component | Latency |
|-----------|---------|
| DETIC detection | 50ms |
| DINOv2-B encoding | 25ms |
| RoI pooling | 2ms |
| MultiModalGNN | 20ms |
| **Total** | **~97ms** |

Note: Both require DETIC for object localization. Option C trades depth estimation for vision encoding.

---

## Evaluation Metrics

### Predicate Detection
- Precision, Recall, F1 per predicate
- Macro-averaged F1 across all predicates
- Confusion matrix for interaction predicates

### Latency
- Mean, P50, P95, P99
- Component-wise breakdown
- First-call warmup vs steady-state

### Generalization
- Train on cups → test on bowls (novel object)
- Train on ALOHA → test on RLBench (novel domain)

---

## Open Questions

1. [ ] LeRobot ALOHA camera intrinsics — where to find?
2. [ ] DETIC vs GroundingDINO — which has better open-vocab?
3. [ ] Memory constraints with DINOv2-B on RTX 500 (4GB)?

---

## Dependencies to Add

```toml
# pyproject.toml
detectron2 = ">=0.6"
transformers = ">=4.35"
timm = ">=0.9"
```

---

## Training Compute Estimates (Moderate Scope)

### Option A: Geometric Fusion
| Stage | Dataset | Epochs | Time (RTX 500) | Time (RTX 4080) |
|-------|---------|--------|----------------|-----------------|
| Fine-tune RelationalGNN | ALOHA 55k | 50 | ~30 min | ~10 min |
| Fine-tune RelationalGNN | RLBench 25k | 50 | ~15 min | ~5 min |
| **Total** | — | — | **~45 min** | **~15 min** |

### Option C: Multi-Modal GNN  
| Stage | Dataset | Epochs | Time (RTX 500) | Time (RTX 4080) |
|-------|---------|--------|----------------|-----------------|
| Train MultiModalGNN | ALOHA 55k | 100 | ~3 hrs | ~45 min |
| Train MultiModalGNN | RLBench 25k | 100 | ~1.5 hrs | ~25 min |
| **Total** | — | — | **~4.5 hrs** | **~1.5 hrs** |

### Recommendation
- **Development/debugging:** RTX 500, small subset (5k frames)
- **Full training:** RTX 4080, full datasets
- **Overnight run:** Option C on RTX 4080 (~2 hrs total)

---

## Notes

- Keep RTX 500 (4GB) as primary development target
- RTX 4080 (16GB) available for full training runs
- All models should support CPU fallback for CI/testing
- RLBench requires X server or headless rendering (EGL/OSMesa)

