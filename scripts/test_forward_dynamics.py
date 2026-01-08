#!/usr/bin/env python3
"""Test the ForwardDynamicsModel and simulate_action MCP tool.

Usage:
    python scripts/test_forward_dynamics.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from gnn_reasoner.data_manager import DataManager
from gnn_reasoner.lerobot_transformer import LeRobotGraphTransformer, ALOHA_KINEMATIC_CHAIN
from gnn_reasoner.model.forward_dynamics import ForwardDynamicsModel
from mcp_ros2_bridge.tools.prediction import PredictionToolsManager


def main():
    print("=" * 60)
    print("ForwardDynamicsModel Test")
    print("=" * 60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load checkpoint
    checkpoint_path = Path("experiments/remote_training/forward_dynamics_e2e/best_model.pt")
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Run training first: ./scripts/remote_train.sh train-fwd")
        return 1
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    model = ForwardDynamicsModel(
        gnn_encoder=None,  # Uses simple encoder (end-to-end trained)
        hidden_dim=checkpoint.get("config", {}).get("hidden_dim", 128),
        action_dim=14,
        num_nodes=16,
        freeze_encoder=False,
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded (epoch {checkpoint.get('epoch', '?')})")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Delta error: {checkpoint.get('val_delta_error', 'N/A'):.4f}")
    
    # Load dataset
    print("\nLoading LeRobot dataset...")
    data_manager = DataManager(repo_id="lerobot/aloha_static_coffee")
    graph_transformer = LeRobotGraphTransformer(
        kinematic_chain=ALOHA_KINEMATIC_CHAIN,
        num_joints=16,
    )
    
    print(f"Dataset loaded: {len(data_manager)} frames")
    
    # Test 1: Direct model inference
    print("\n" + "-" * 40)
    print("Test 1: Direct Model Inference")
    print("-" * 40)
    
    # Get a sample frame
    frame = data_manager.get_frame(100)
    state = frame["observation.state"]
    action = frame["action"]
    
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action, dtype=torch.float32)
    
    # Build graph
    graph = graph_transformer.to_graph(state).to(device)
    action = action.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(graph, action)
    
    print(f"  Delta shape: {outputs['delta'].shape}")
    print(f"  Confidence: {outputs['confidence'].item():.3f}")
    print(f"  Feasibility logit: {outputs['feasibility'].item():.3f}")
    print(f"  Uncertainty shape: {outputs['uncertainty'].shape}")
    
    # Test 2: Simulate single action
    print("\n" + "-" * 40)
    print("Test 2: Simulate Single Action")
    print("-" * 40)
    
    result = model.simulate(graph, [action], return_trajectory=True)
    print(f"  Results type: {type(result)}")
    if isinstance(result, list):
        result = result[0]
    print(f"  Predicted positions shape: {result.predicted_positions.shape}")
    print(f"  Position delta shape: {result.position_delta.shape}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Is feasible: {result.is_feasible}")
    
    # Test 3: Simulate multi-step sequence
    print("\n" + "-" * 40)
    print("Test 3: Simulate Multi-Step Sequence (5 steps)")
    print("-" * 40)
    
    # Get 5 consecutive actions
    actions = []
    for i in range(5):
        frame = data_manager.get_frame(100 + i)
        a = frame["action"]
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float32)
        actions.append(a.to(device))
    
    results = model.simulate(graph, actions, return_trajectory=True)
    print(f"  Number of results: {len(results)}")
    
    for i, r in enumerate(results):
        print(f"  Step {i+1}: confidence={r.confidence:.3f}, feasible={r.is_feasible}")
    
    # Test 4: MCP Response format
    print("\n" + "-" * 40)
    print("Test 4: MCP Response Format")
    print("-" * 40)
    
    mcp_response = model.to_mcp_response(results)
    print(f"  Recommendation: {mcp_response['recommendation']}")
    print(f"  Min confidence: {mcp_response['min_confidence']:.3f}")
    print(f"  Overall feasible: {mcp_response['overall_feasible']}")
    print(f"  Num steps: {mcp_response['num_steps']}")
    
    # Test 5: PredictionToolsManager simulate_action
    print("\n" + "-" * 40)
    print("Test 5: PredictionToolsManager.simulate_action()")
    print("-" * 40)
    
    from gnn_reasoner.model.relational_gnn import RelationalGNN
    
    # Create a dummy GNN (not used, just for the manager)
    gnn_model = RelationalGNN(hidden_dim=128, num_predicates=9).to(device)
    
    tools_manager = PredictionToolsManager(
        data_manager=data_manager,
        graph_transformer=graph_transformer,
        gnn_model=gnn_model,
        forward_model=model,
    )
    
    # Set frame
    data_manager._current_frame_idx = 100
    
    # Test simulate_action with num_steps
    result = tools_manager.simulate_action(
        action_sequence=None,
        num_steps=3,
        confidence_threshold=0.5,
    )
    
    print(f"  Recommendation: {result.get('recommendation', 'N/A')}")
    print(f"  Min confidence: {result.get('min_confidence', 'N/A')}")
    print(f"  Overall feasible: {result.get('overall_feasible', 'N/A')}")
    print(f"  Inference time: {result.get('inference_time_ms', 'N/A')} ms")
    
    # Test with custom action sequence
    print("\n  Testing with custom action sequence...")
    custom_actions = [
        [0.1] * 14,  # Small joint velocities
        [0.0] * 14,  # Zero action
        [-0.1] * 14,  # Reverse
    ]
    
    result = tools_manager.simulate_action(
        action_sequence=custom_actions,
        confidence_threshold=0.5,
    )
    
    print(f"  Recommendation: {result.get('recommendation', 'N/A')}")
    print(f"  Min confidence: {result.get('min_confidence', 'N/A'):.3f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

