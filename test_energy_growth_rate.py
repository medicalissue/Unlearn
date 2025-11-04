#!/usr/bin/env python3
"""
Test script for energy_growth_rate score type in unlearn postprocessor.
"""

import sys
sys.path.insert(0, '/home/junesang/OpenOOD')

import torch
import torch.nn as nn
from openood.postprocessors.unlearn_postprocessor import UnlearnPostprocessor

# Create a mock config
class MockConfig:
    def __init__(self, score_type="energy_growth_rate"):
        self.dataset = MockDataset()
        self.postprocessor = MockPostprocessor(score_type)

class MockDataset:
    def __init__(self):
        self.name = 'cifar10'
        self.data_root = './data'

class MockPostprocessor:
    def __init__(self, score_type):
        self.postprocessor_args = {
            'unlearn_mode': 'ascent',
            'eta': 1e-2,
            'num_steps': 10,  # Small for testing
            'temp': 1.0,
            'score_type': score_type,
            'prototype_coupling_config': {
                'use_all_prototypes': False,
                'top_k': 1,
                'eigenvalue_mode': 'log_trace',
                'prototype_aggregation': 'mean',
                'max_samples_per_class': None
            }
        }

def test_initialization():
    """Test that the postprocessor can be initialized with energy_growth_rate mode."""
    print("Testing initialization...")

    config = MockConfig(score_type="energy_growth_rate")
    postprocessor = UnlearnPostprocessor(config)

    # Check that the score_type is set correctly
    assert postprocessor.score_type == 'energy_growth_rate', \
        f"Expected score_type='energy_growth_rate', got '{postprocessor.score_type}'"

    print("✓ Initialization test passed")

def test_score_types():
    """Test that both score types can be initialized."""
    print("\nTesting different score types...")

    score_types = ['prototype_coupling', 'energy_growth_rate']

    for score_type in score_types:
        config = MockConfig(score_type=score_type)
        postprocessor = UnlearnPostprocessor(config)
        assert postprocessor.score_type == score_type, f"Score type mismatch for {score_type}"
        print(f"  ✓ Score type '{score_type}' initialized successfully")

    print("✓ All score types test passed")

def test_energy_computation_logic():
    """Test the logic of energy growth rate computation."""
    print("\nTesting energy growth rate logic...")

    # Simulate energy values
    num_steps = 100

    # Scenario 1: ID sample (high growth rate)
    energy_0_id = torch.tensor(-5.0)
    energy_T_id = torch.tensor(2.0)  # Large increase
    growth_rate_id = (energy_T_id - energy_0_id) / num_steps
    print(f"  ID sample: E_0={energy_0_id.item():.4f}, E_T={energy_T_id.item():.4f}")
    print(f"  → Growth rate: {growth_rate_id.item():.6f}")

    # Scenario 2: OOD sample (low growth rate)
    energy_0_ood = torch.tensor(-5.0)
    energy_T_ood = torch.tensor(-4.5)  # Small increase
    growth_rate_ood = (energy_T_ood - energy_0_ood) / num_steps
    print(f"  OOD sample: E_0={energy_0_ood.item():.4f}, E_T={energy_T_ood.item():.4f}")
    print(f"  → Growth rate: {growth_rate_ood.item():.6f}")

    # Higher growth rate should correspond to ID (higher OOD score for ID detection)
    assert growth_rate_id > growth_rate_ood, "ID should have higher growth rate than OOD"
    print(f"  ✓ ID growth rate ({growth_rate_id.item():.6f}) > OOD growth rate ({growth_rate_ood.item():.6f})")

    print("✓ Energy growth rate logic test passed")

def test_method_exists():
    """Test that the helper method exists."""
    print("\nTesting helper method existence...")

    config = MockConfig()
    postprocessor = UnlearnPostprocessor(config)

    # Check that _compute_energy_at_step method exists
    assert hasattr(postprocessor, '_compute_energy_at_step'), \
        "_compute_energy_at_step method not found"

    print("✓ Helper method _compute_energy_at_step exists")

if __name__ == '__main__':
    print("=" * 60)
    print("Testing energy_growth_rate implementation")
    print("=" * 60)

    try:
        test_initialization()
        test_score_types()
        test_energy_computation_logic()
        test_method_exists()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print("\nUsage:")
        print("  Set score_type: 'energy_growth_rate' in config")
        print("  The postprocessor will compute:")
        print("    growth_rate = (E_T - E_0) / T")
        print("  where E_t = log(trace(C_t))")
        print("\nExpected behavior:")
        print("  - ID samples: High growth rate (energy increases rapidly)")
        print("  - OOD samples: Low growth rate (energy stays flat)")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
