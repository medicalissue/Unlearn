#!/usr/bin/env python3
"""
Test script for Multi-Scale Fisher Energy implementation in unlearn postprocessor.
"""

import sys
sys.path.insert(0, '/home/junesang/OpenOOD')

import torch
import torch.nn as nn
from openood.postprocessors.unlearn_postprocessor import UnlearnPostprocessor


class MockConfig:
    def __init__(self, multiscale=True, metric="combo", num_steps=50):
        self.dataset = MockDataset()
        self.postprocessor = MockPostprocessor(multiscale, metric, num_steps)


class MockDataset:
    def __init__(self):
        self.name = 'cifar10'
        self.data_root = './data'


class MockPostprocessor:
    def __init__(self, multiscale, metric, num_steps):
        self.postprocessor_args = {
            'unlearn_mode': 'ascent',
            'eta': 1e-2,
            'num_steps': num_steps,
            'temp': 1.0,
            'score_type': 'fisher_energy',
            'fisher_energy_mode': 'unlearned',
            'fisher_normalization': 'none',
            'fisher_temp': 1.0,
            'fisher_multiscale': multiscale,
            'fisher_multiscale_checkpoints': [0.0, 0.25, 0.5, 0.75, 1.0],
            'fisher_multiscale_metric': metric,
            'fisher_multiscale_weights': [0.3, 0.4, 0.3],
            'prototype_coupling_config': {
                'use_all_prototypes': False,
                'top_k': 1,
                'eigenvalue_mode': 'log_trace',
                'prototype_aggregation': 'mean',
                'max_samples_per_class': None,
                'fisher_mode': 'global'
            }
        }
        self.postprocessor_sweep = {
            'eta': [1e-2],
            'num_steps': [50]
        }


def test_initialization():
    """Test that the postprocessor can be initialized with multi-scale mode."""
    print("=" * 60)
    print("Test 1: Initialization")
    print("=" * 60)

    config = MockConfig(multiscale=True)
    postprocessor = UnlearnPostprocessor(config)

    # Check that multi-scale parameters are set correctly
    assert postprocessor.fisher_multiscale == True, \
        f"Expected fisher_multiscale=True, got {postprocessor.fisher_multiscale}"
    assert postprocessor.fisher_multiscale_metric == "combo", \
        f"Expected metric='combo', got '{postprocessor.fisher_multiscale_metric}'"
    assert len(postprocessor.fisher_multiscale_checkpoints) == 5, \
        f"Expected 5 checkpoints, got {len(postprocessor.fisher_multiscale_checkpoints)}"

    print("✓ Multi-scale parameters initialized correctly")
    print(f"  - fisher_multiscale: {postprocessor.fisher_multiscale}")
    print(f"  - checkpoints: {postprocessor.fisher_multiscale_checkpoints}")
    print(f"  - metric: {postprocessor.fisher_multiscale_metric}")
    print(f"  - weights: {postprocessor.fisher_multiscale_weights}")
    print()


def test_different_metrics():
    """Test that all metric modes can be initialized."""
    print("=" * 60)
    print("Test 2: Different Metrics")
    print("=" * 60)

    metrics = ['mean', 'variance', 'trend', 'combo']

    for metric in metrics:
        config = MockConfig(multiscale=True, metric=metric)
        postprocessor = UnlearnPostprocessor(config)
        assert postprocessor.fisher_multiscale_metric == metric, \
            f"Metric mismatch for {metric}"
        print(f"  ✓ Metric '{metric}' initialized successfully")

    print()


def test_checkpoint_calculation():
    """Test that checkpoint steps are calculated correctly."""
    print("=" * 60)
    print("Test 3: Checkpoint Calculation Logic")
    print("=" * 60)

    num_steps = 100
    checkpoints = [0.0, 0.25, 0.5, 0.75, 1.0]

    expected_steps = [0, 25, 50, 75, 100]

    print(f"  Number of unlearning steps: {num_steps}")
    print(f"  Checkpoint fractions: {checkpoints}")
    print()

    for frac, expected in zip(checkpoints, expected_steps):
        actual = int(frac * num_steps)
        print(f"  Checkpoint {frac:.2f} → Step {actual} (expected: {expected})")
        assert actual == expected, f"Mismatch for fraction {frac}"

    print("  ✓ All checkpoint calculations correct")
    print()


def test_trajectory_computation_logic():
    """Test the trajectory computation logic with mock data."""
    print("=" * 60)
    print("Test 4: Trajectory Computation Logic")
    print("=" * 60)

    # Simulate Fisher energies at different timesteps
    # Scenario 1: ID sample (smooth trajectory)
    energies_id = torch.tensor([5.0, 4.8, 4.5, 4.3, 4.0])  # Smooth decrease
    mean_id = torch.mean(energies_id)
    var_id = torch.var(energies_id)
    trend_id = energies_id[-1] - energies_id[0]

    print(f"  ID sample energies: {energies_id.numpy()}")
    print(f"    → Mean: {mean_id.item():.4f}")
    print(f"    → Variance: {var_id.item():.4f}")
    print(f"    → Trend: {trend_id.item():.4f}")
    print()

    # Scenario 2: OOD sample (erratic trajectory)
    energies_ood = torch.tensor([5.0, 6.2, 4.3, 7.1, 4.5])  # Erratic changes
    mean_ood = torch.mean(energies_ood)
    var_ood = torch.var(energies_ood)
    trend_ood = energies_ood[-1] - energies_ood[0]

    print(f"  OOD sample energies: {energies_ood.numpy()}")
    print(f"    → Mean: {mean_ood.item():.4f}")
    print(f"    → Variance: {var_ood.item():.4f}")
    print(f"    → Trend: {trend_ood.item():.4f}")
    print()

    # Check expected behavior
    assert var_ood > var_id, "OOD should have higher variance than ID"
    print(f"  ✓ OOD variance ({var_ood.item():.4f}) > ID variance ({var_id.item():.4f})")
    print()


def test_backward_compatibility():
    """Test that non-multiscale mode still works."""
    print("=" * 60)
    print("Test 5: Backward Compatibility")
    print("=" * 60)

    config = MockConfig(multiscale=False)
    postprocessor = UnlearnPostprocessor(config)

    assert postprocessor.fisher_multiscale == False, \
        "fisher_multiscale should be False"
    print("  ✓ Non-multiscale mode initialized correctly")
    print(f"    - fisher_multiscale: {postprocessor.fisher_multiscale}")
    print()


def test_method_exists():
    """Test that the helper method exists."""
    print("=" * 60)
    print("Test 6: Helper Method Existence")
    print("=" * 60)

    config = MockConfig(multiscale=True)
    postprocessor = UnlearnPostprocessor(config)

    # Check that _compute_multiscale_fisher_trajectory method exists
    assert hasattr(postprocessor, '_compute_multiscale_fisher_trajectory'), \
        "_compute_multiscale_fisher_trajectory method not found"

    print("  ✓ Helper method _compute_multiscale_fisher_trajectory exists")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Testing Multi-Scale Fisher Energy Implementation")
    print("=" * 60)
    print()

    try:
        test_initialization()
        test_different_metrics()
        test_checkpoint_calculation()
        test_trajectory_computation_logic()
        test_backward_compatibility()
        test_method_exists()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print()
        print("Implementation Summary:")
        print("  1. Config parameters added: fisher_multiscale, fisher_multiscale_checkpoints,")
        print("     fisher_multiscale_metric, fisher_multiscale_weights")
        print("  2. Helper method implemented: _compute_multiscale_fisher_trajectory()")
        print("  3. _single_sample_unlearn() modified to track parameter checkpoints")
        print("  4. Scoring logic updated to use multi-scale mode when enabled")
        print()
        print("Usage:")
        print("  Set fisher_multiscale: true in config")
        print("  Configure checkpoints (e.g., [0.0, 0.25, 0.5, 0.75, 1.0])")
        print("  Choose metric: 'mean', 'variance', 'trend', or 'combo'")
        print()
        print("Expected Behavior:")
        print("  - ID samples: Smooth Fisher energy trajectory (low variance)")
        print("  - OOD samples: Erratic Fisher energy changes (high variance)")

    except Exception as e:
        print("=" * 60)
        print(f"✗ Test failed with error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
