"""Test script to verify APS filtering based on component_weights"""

# Mock config object
class MockConfig:
    def __init__(self):
        self.dataset = type('obj', (object,), {'name': 'cifar10'})()
        self.postprocessor = type('obj', (object,), {
            'postprocessor_args': {
                'eta': 1e-3,
                'num_steps': 1,
                'temp': 1.0,
                'use_gradnorm': False,
                'use_feature_grad': True,
                'unlearn_mode': 'ascent',
                'score_type': 'feature_aware',
                'feature_aware_config': {
                    'mode': 'baseline',
                    'component_weights': {
                        'feature_norm': 0.0,   # NOT USED
                        'distance': 0.0,        # NOT USED
                        'weight_shift': 1.0,    # USED
                    },
                    'distance_metric': 'l1',
                    'fractional_p': 0.5,
                },
            },
            'postprocessor_sweep': {
                'eta': [1e-3, 5e-3],
                'num_steps': [1, 3],
                'use_feature_grad': [True],
                'score_type': ['feature_aware'],
                'feature_aware_config': {
                    'mode': ['baseline', 'weighted'],
                    'component_weights': {
                        'feature_norm': [1.0, 2.0],
                        'distance': [1.0, 2.0],
                        'weight_shift': [1.0],
                    },
                    'distance_metric': ['l1', 'l2', 'fractional'],
                    'fractional_p': [0.3, 0.5],
                    'angular_weight': [0.5],
                },
            },
        })()

# Test
config = MockConfig()

print("="*70)
print("Testing APS Filtering with component_weights")
print("="*70)
print("\nConfig component_weights:")
print(f"  feature_norm: {config.postprocessor.postprocessor_args['feature_aware_config']['component_weights']['feature_norm']}")
print(f"  distance: {config.postprocessor.postprocessor_args['feature_aware_config']['component_weights']['distance']}")
print(f"  weight_shift: {config.postprocessor.postprocessor_args['feature_aware_config']['component_weights']['weight_shift']}")

print("\n" + "="*70)
print("BEFORE filtering - postprocessor_sweep keys:")
print("="*70)
def print_keys(d, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print("  " * indent + f"- {k}:")
            print_keys(v, indent+1)
        else:
            value_preview = str(v)[:50] + "..." if len(str(v)) > 50 else str(v)
            print("  " * indent + f"- {k}: {value_preview}")

print_keys(config.postprocessor.postprocessor_sweep)

# Import and create postprocessor
import sys
sys.path.insert(0, '/home/junesang/OpenOOD')

# Direct import to avoid faiss dependency
import importlib.util
spec = importlib.util.spec_from_file_location(
    "unlearn_postprocessor",
    "/home/junesang/OpenOOD/openood/postprocessors/unlearn_postprocessor.py"
)
unlearn_module = importlib.util.module_from_spec(spec)
sys.modules['unlearn_postprocessor'] = unlearn_module

# Mock base_postprocessor and info
class BasePostprocessor:
    def __init__(self, config):
        self.config = config
        self.APS_mode = False
        self.hyperparam_search_done = False

num_classes_dict = {'cifar10': 10}

# Inject mocks
sys.modules['openood.postprocessors.base_postprocessor'] = type('module', (), {'BasePostprocessor': BasePostprocessor})()
sys.modules['openood.postprocessors.info'] = type('module', (), {'num_classes_dict': num_classes_dict})()

spec.loader.exec_module(unlearn_module)
UnlearnPostprocessor = unlearn_module.UnlearnPostprocessor

print("\n" + "="*70)
print("Creating UnlearnPostprocessor (filtering will happen)...")
print("="*70)
postproc = UnlearnPostprocessor(config)

print("\n" + "="*70)
print("AFTER filtering - args_dict (flattened):")
print("="*70)
for i, (key, value) in enumerate(postproc.args_dict.items(), 1):
    value_preview = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
    print(f"{i:3d}. {key:60s} = {value_preview}")

print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

# Check if distance-related keys are removed
distance_keys = [
    'feature_aware_config.distance_metric',
    'feature_aware_config.fractional_p',
    'feature_aware_config.angular_weight',
]

removed_count = 0
kept_count = 0

for key in distance_keys:
    if key in postproc.args_dict:
        print(f"❌ FAILED: {key} should be removed but found in args_dict")
        kept_count += 1
    else:
        print(f"✓ PASSED: {key} successfully removed")
        removed_count += 1

# Check if essential keys are kept
essential_keys = [
    'eta',
    'num_steps',
    'use_feature_grad',
    'score_type',
    'feature_aware_config.mode',
    'feature_aware_config.component_weights.weight_shift',
]

for key in essential_keys:
    if key in postproc.args_dict:
        print(f"✓ PASSED: {key} is kept")
        kept_count += 1
    else:
        print(f"❌ FAILED: {key} should be kept but not found")
        removed_count += 1

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
total_before = len(list(config.postprocessor.postprocessor_sweep.keys()))
# Count nested keys recursively
def count_all_keys(d):
    count = 0
    for v in d.values():
        if isinstance(v, dict):
            count += count_all_keys(v)
        else:
            count += 1
    return count

total_before_flat = count_all_keys(config.postprocessor.postprocessor_sweep)
total_after = len(postproc.args_dict)

print(f"Total hyperparameters before filtering: {total_before_flat}")
print(f"Total hyperparameters after filtering:  {total_after}")
print(f"Reduction: {total_before_flat - total_after} parameters removed")
print(f"\nEstimated search space reduction:")
print(f"  Before: ~{2**total_before_flat:,} combinations (assuming 2 values each)")
print(f"  After:  ~{2**total_after:,} combinations")
print(f"  Speedup: ~{2**(total_before_flat - total_after):.1f}x faster")

print("\n" + "="*70)
print("Test completed!")
print("="*70)
