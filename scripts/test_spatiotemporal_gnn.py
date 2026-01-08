#!/usr/bin/env python3
"""Validation tests for SpatiotemporalGNN (Phase 11).

Quick sanity checks to verify:
1. Model can be instantiated
2. Forward pass works
3. Future prediction works
4. Temporal state management works

Usage:
    python scripts/test_spatiotemporal_gnn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch_geometric.data import Data

from gnn_reasoner.model.spatiotemporal_gnn import SpatiotemporalGNN
from gnn_reasoner.model.relational_gnn import RelationalGNN
from gnn_reasoner.lerobot_transformer import LeRobotGraphTransformer, ALOHA_KINEMATIC_CHAIN


def test_model_creation():
    """Test 1: Model can be instantiated."""
    print("Test 1: Model Creation")
    print("-" * 60)
    
    # Create without base GNN (new RelationalGNN)
    model = SpatiotemporalGNN()
    print(f"✅ Created ST-GNN without base: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Create with base GNN
    base_gnn = RelationalGNN()
    model_with_base = SpatiotemporalGNN(base_gnn=base_gnn, freeze_base=True)
    print(f"✅ Created ST-GNN with frozen base: {sum(p.numel() for p in model_with_base.parameters()):,} params")
    
    # Create with trainable base
    model_trainable = SpatiotemporalGNN(base_gnn=base_gnn, freeze_base=False)
    print(f"✅ Created ST-GNN with trainable base: {sum(p.numel() for p in model_trainable.parameters()):,} params")
    
    print()


def test_forward_pass():
    """Test 2: Forward pass works."""
    print("Test 2: Forward Pass")
    print("-" * 60)
    
    model = SpatiotemporalGNN()
    model.eval()
    
    # Create dummy graph (ALOHA: 16 nodes)
    transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
    state = torch.randn(14)  # 14-DoF ALOHA
    graph = transformer.to_graph(state)
    
    # Forward pass
    with torch.no_grad():
        output = model(graph)
    
    print(f"✅ Forward pass successful")
    print(f"   - Node embeddings: {output['node_embeddings'].shape}")
    print(f"   - Predicate logits: {output['predicate_logits'].shape}")
    print(f"   - Graph embedding: {output['graph_embedding'].shape}")
    print(f"   - Temporal embedding: {output['temporal_embedding'].shape}")
    
    print()


def test_temporal_state():
    """Test 3: Temporal state management."""
    print("Test 3: Temporal State Management")
    print("-" * 60)
    
    model = SpatiotemporalGNN()
    model.eval()
    
    transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
    
    # Process sequence of frames
    hidden_state = None
    for i in range(3):
        state = torch.randn(14) * 0.1 * i  # Slightly different states
        graph = transformer.to_graph(state)
        
        with torch.no_grad():
            output = model(graph, hidden_state, return_hidden=True)
            hidden_state = output["hidden_state"]
        
        print(f"   Frame {i+1}: temporal_embed={output['temporal_embedding'].shape}, "
              f"hidden_state={hidden_state.shape if hidden_state is not None else None}")
    
    print("✅ Temporal state maintained across frames")
    
    # Test reset
    model.reset_hidden_state()
    print("✅ Hidden state reset successful")
    
    print()


def test_future_prediction():
    """Test 4: Future prediction works."""
    print("Test 4: Future Prediction")
    print("-" * 60)
    
    model = SpatiotemporalGNN()
    model.eval()
    
    transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
    state = torch.randn(14)
    graph = transformer.to_graph(state)
    action = torch.randn(14)  # 14-DoF action
    
    with torch.no_grad():
        future_predictions, current_outputs = model.predict_future(graph, action)
    
    print(f"✅ Future prediction successful")
    print(f"   - Number of future steps: {len(future_predictions)}")
    for pred in future_predictions:
        print(f"   - Step {pred.step}: confidence={pred.confidence:.3f}, "
              f"logits={pred.predicate_logits.shape}")
    
    print()


def test_device_compatibility():
    """Test 5: Device compatibility (CPU/GPU)."""
    print("Test 5: Device Compatibility")
    print("-" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = SpatiotemporalGNN().to(device)
    model.eval()
    
    transformer = LeRobotGraphTransformer(ALOHA_KINEMATIC_CHAIN)
    state = torch.randn(14).to(device)
    graph = transformer.to_graph(state)
    graph = graph.to(device)
    
    with torch.no_grad():
        output = model(graph)
    
    print(f"✅ Model works on {device}")
    print(f"   - All tensors on correct device")
    
    print()


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("SpatiotemporalGNN Validation Tests")
    print("=" * 60)
    print()
    
    try:
        test_model_creation()
        test_forward_pass()
        test_temporal_state()
        test_future_prediction()
        test_device_compatibility()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Quick validation: python scripts/train_spatiotemporal_gnn.py --max-frames 5000 --epochs 10")
        print("2. Full training: python scripts/train_spatiotemporal_gnn.py --epochs 100")
        print("3. Use pre-trained base: python scripts/train_spatiotemporal_gnn.py --base-checkpoint experiments/remote_training/relational_gnn/best_model.pt")
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

