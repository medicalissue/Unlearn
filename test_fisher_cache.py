"""Test script to verify Fisher matrix caching fixes."""
import torch
import torch.nn as nn
import sys
import os

# Add OpenOOD to path
sys.path.insert(0, '/home/junesang/OpenOOD')

from openood.postprocessors.unlearn_postprocessor import UnlearnPostprocessor
from openood.utils import Config

# Create a minimal config
config = Config({
    'postprocessor': {
        'name': 'unlearn',
        'postprocessor_args': {
            'fisher_power': 9.0,
            'fisher_mode': 'global'
        }
    },
    'dataset': {
        'name': 'cifar10'
    },
    'network': {
        'checkpoint': '/home/junesang/OpenOOD/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
    }
})

print("✓ Config created")

# Initialize postprocessor
postprocessor = UnlearnPostprocessor(config)
print(f"✓ Postprocessor initialized")
print(f"  - fisher_power: {postprocessor.fisher_power}")
print(f"  - fisher_mode: {postprocessor.fisher_mode}")

# Test cache path generation
try:
    cache_path = postprocessor._get_fisher_cache_path()
    print(f"✓ Cache path generated: {cache_path}")
except Exception as e:
    print(f"✗ Cache path generation failed: {e}")
    sys.exit(1)

# Test with config.ckpt_path (simulating CLI argument)
config2 = Config({
    'postprocessor': {
        'name': 'unlearn',
        'postprocessor_args': {
            'fisher_power': 9.0,
            'fisher_mode': 'global'
        }
    },
    'dataset': {
        'name': 'cifar10'
    },
    'ckpt_path': '/home/junesang/OpenOOD/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
})

postprocessor2 = UnlearnPostprocessor(config2)
try:
    cache_path2 = postprocessor2._get_fisher_cache_path()
    print(f"✓ Cache path from ckpt_path: {cache_path2}")
except Exception as e:
    print(f"✗ Cache path from ckpt_path failed: {e}")
    sys.exit(1)

# Create a dummy network with FC layer
class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 10, bias=True)

    def forward(self, x, return_feature=False):
        if return_feature:
            return torch.randn(x.shape[0], 512).cuda()
        return self.fc(torch.randn(x.shape[0], 512).cuda())

net = DummyNet().cuda()
print("✓ Dummy network created")

# Create dummy data loader
class DummyDataset:
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        for i in range(self.size):
            yield {
                'data': torch.randn(8, 3, 32, 32).cuda(),
                'label': torch.randint(0, 10, (8,)).cuda()
            }

id_loader_dict = {'train': DummyDataset(size=5)}
print("✓ Dummy data loader created")

# Test Fisher matrix computation (should work with vmap fix)
print("\nTesting Fisher matrix computation...")
try:
    postprocessor.setup(net, id_loader_dict, {})
    print("✓ Fisher matrix computed successfully")
    print(f"  - Fisher W shape: {postprocessor.fisher_W_tensor.shape}")
    print(f"  - Fisher b shape: {postprocessor.fisher_b_tensor.shape if postprocessor.fisher_b_tensor is not None else 'None'}")
except Exception as e:
    print(f"✗ Fisher matrix computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test caching
print("\nTesting Fisher matrix caching...")
try:
    # Save cache
    postprocessor._save_fisher_matrix(cache_path)
    print(f"✓ Fisher matrix saved to: {cache_path}")

    # Load cache
    postprocessor3 = UnlearnPostprocessor(config)
    success = postprocessor3._load_fisher_matrix(cache_path)
    if success:
        print("✓ Fisher matrix loaded from cache")
        print(f"  - Fisher W shape: {postprocessor3.fisher_W_tensor.shape}")
    else:
        print("✗ Failed to load Fisher matrix from cache")
        sys.exit(1)
except Exception as e:
    print(f"✗ Caching test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("All tests passed! ✓")
print("="*50)
