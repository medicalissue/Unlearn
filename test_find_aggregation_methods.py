#!/usr/bin/env python3
"""Test script to verify all Fisher aggregation methods work correctly."""

import torch
import torch.nn as nn
from openood.postprocessors.find_postprocessor import FInDPostprocessor

class SimpleConfig:
    """Mock config for testing."""
    def __init__(self, aggregation_method):
        self.dataset = type('obj', (object,), {'name': 'cifar10'})()
        self.postprocessor = type('obj', (object,), {
            'postprocessor_args': {
                'fisher_power': 1.0,
                'use_adaptive_power': False,
                'adaptive_metric': 'entropy',
                'adaptive_alpha': 1.0,
                'aggregation_method': aggregation_method,
                # Method-specific params
                'sep_threshold_percentile': 50.0,
                'sep_w_small': 1.0,
                'sep_w_large': 1.0,
                'sep_large_power': 1.0,
                'ms_percentiles': [90, 50, 10],
                'ms_weights': [1.0, 1.0, 1.0],
                'norm_num_groups': 5,
                'bayes_uncertainty_weight': 1.0,
                'bayes_confidence_weight': 1.0,
                'dual_alpha': 0.5,
                'dual_beta': 0.5,
                'dual_aggregator': 'median',
                'aw_temperature': 1.0,
            },
            'postprocessor_sweep': {}
        })()

def test_aggregation_method(method_name):
    """Test a single aggregation method."""
    print(f"\n{'='*60}")
    print(f"Testing: {method_name}")
    print(f"{'='*60}")

    try:
        # Create config and postprocessor
        config = SimpleConfig(method_name)
        postprocessor = FInDPostprocessor(config)

        # Create mock data
        batch_size = 8
        dim = 100

        g_batch = torch.randn(batch_size, dim).cuda()
        F_vec = torch.abs(torch.randn(dim)).cuda() + 0.01  # Ensure positive
        fisher_power = torch.ones(batch_size).cuda()

        # Test the aggregation
        with torch.no_grad():
            score = postprocessor._compute_fisher_score(g_batch, F_vec, fisher_power)

        print(f"✓ Success!")
        print(f"  Input shape: g={g_batch.shape}, F={F_vec.shape}")
        print(f"  Output shape: {score.shape}")
        print(f"  Score range: [{score.min().item():.4f}, {score.max().item():.4f}]")
        print(f"  Score mean: {score.mean().item():.4f}")

        return True

    except Exception as e:
        print(f"✗ Failed!")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test all aggregation methods."""
    methods = [
        "logsumexp",
        "separated",
        "multiscale",
        "normalized",
        "bayesian",
        "dualpath",
        "adaptive_weight"
    ]

    print("="*60)
    print("Fisher Aggregation Methods Test")
    print("="*60)

    results = {}
    for method in methods:
        results[method] = test_aggregation_method(method)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")

    for method, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {method}")

    all_passed = all(results.values())
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    print(f"{'='*60}")

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())
