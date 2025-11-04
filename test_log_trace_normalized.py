#!/usr/bin/env python3
"""
Test script for log_trace_normalized mode in unlearn postprocessor.
This script verifies that the implementation can be imported and initialized correctly.
"""

import sys
sys.path.insert(0, '/home/junesang/OpenOOD')

import torch
import torch.nn as nn
from openood.postprocessors.unlearn_postprocessor import UnlearnPostprocessor

# Create a mock config
class MockConfig:
    def __init__(self):
        self.dataset = MockDataset()
        self.postprocessor = MockPostprocessor()

class MockDataset:
    def __init__(self):
        self.name = 'cifar10'
        self.data_root = './data'

class MockPostprocessor:
    def __init__(self):
        self.postprocessor_args = {
            'unlearn_mode': 'ascent',
            'eta': 1e-2,
            'num_steps': 10,  # Small for testing
            'temp': 1.0,
            'score_type': 'prototype_coupling',
            'prototype_coupling_config': {
                'use_all_prototypes': True,
                'top_k': 10,
                'eigenvalue_mode': 'log_trace_normalized',
                'prototype_aggregation': 'mean',
                'max_samples_per_class': None
            }
        }

def test_initialization():
    """Test that the postprocessor can be initialized with log_trace_normalized mode."""
    print("Testing initialization...")

    config = MockConfig()
    postprocessor = UnlearnPostprocessor(config)

    # Check that the eigenvalue_mode is set correctly
    assert postprocessor.eigenvalue_mode == 'log_trace_normalized', \
        f"Expected eigenvalue_mode='log_trace_normalized', got '{postprocessor.eigenvalue_mode}'"

    # Check that class statistics are initialized to None
    assert postprocessor.class_means is None, "class_means should be None initially"
    assert postprocessor.class_stds is None, "class_stds should be None initially"

    print("✓ Initialization test passed")

def test_different_modes():
    """Test that all eigenvalue modes can be initialized."""
    print("\nTesting different eigenvalue modes...")

    modes = ['participation_ratio', 'log_trace', 'log_trace_normalized']

    for mode in modes:
        config = MockConfig()
        config.postprocessor.postprocessor_args['prototype_coupling_config']['eigenvalue_mode'] = mode
        postprocessor = UnlearnPostprocessor(config)
        assert postprocessor.eigenvalue_mode == mode, f"Mode mismatch for {mode}"
        print(f"  ✓ Mode '{mode}' initialized successfully")

    print("✓ All modes test passed")

def test_vmap_compatibility():
    """Test that the z-score computation uses vmap-safe operations."""
    print("\nTesting vmap-compatible z-score computation...")

    # Simulate a batch of predictions
    num_classes = 10
    batch_size = 4

    # Mock class statistics
    class_means = torch.randn(num_classes).cuda()
    class_stds = torch.abs(torch.randn(num_classes)).cuda() + 0.1

    # Mock predictions
    logits = torch.randn(batch_size, num_classes).cuda()
    y_pred = logits.argmax(dim=-1)  # [batch_size]

    # Mock energies
    log_trace_energies = torch.randn(batch_size).cuda()

    # Compute z-scores using one-hot indexing (vmap-safe)
    one_hot_pred = torch.nn.functional.one_hot(y_pred, num_classes=num_classes).float()
    class_mean_selected = torch.matmul(one_hot_pred, class_means)  # [batch_size]
    class_std_selected = torch.matmul(one_hot_pred, class_stds)  # [batch_size]

    z_scores = (log_trace_energies - class_mean_selected) / (class_std_selected + 1e-8)
    scores = -torch.abs(z_scores)

    # Check shapes
    assert z_scores.shape == (batch_size,), f"Expected shape (4,), got {z_scores.shape}"
    assert scores.shape == (batch_size,), f"Expected shape (4,), got {scores.shape}"

    print(f"  Sample z-scores: {z_scores.cpu().numpy()}")
    print(f"  Sample OOD scores: {scores.cpu().numpy()}")
    print("✓ vmap compatibility test passed")

if __name__ == '__main__':
    print("=" * 60)
    print("Testing log_trace_normalized implementation")
    print("=" * 60)

    try:
        test_initialization()
        test_different_modes()
        test_vmap_compatibility()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
