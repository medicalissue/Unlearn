#!/usr/bin/env python
"""
Test consistency between visualize_fc_weight_spatial.py and FInD postprocessor.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as trn

# Load model and Fisher matrix
from openood.networks import ResNet50

# Load pretrained weights
net = ResNet50(num_classes=1000)
ckpt = torch.load('results/pretrained_weights/resnet50_imagenet1k_v1.pth', map_location='cpu')
net.load_state_dict(ckpt)
net.eval()

# Load Fisher matrix
fisher_cache = torch.load(os.path.expanduser('~/.cache/openood/fisher_matrices/fisher_imagenet_ResNet50_nll.pt'), map_location='cpu')
fisher_W = fisher_cache['fisher_W']

# Load a test image
preprocessor = trn.Compose([
    trn.Resize(256),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_path = 'temp/ID/ILSVRC2012_val_00036290.JPEG'
if not os.path.exists(img_path):
    img_path = 'data/images_largescale/imagenet_1k/val/ILSVRC2012_val_00036290.JPEG'

img = Image.open(img_path).convert('RGB')
img_tensor = preprocessor(img).unsqueeze(0)  # [1, 3, 224, 224]

# Forward pass
with torch.no_grad():
    output = net(img_tensor, return_feature=True)
    if isinstance(output, tuple):
        logits, features = output
    else:
        features = output
        logits = net.fc(features)

# Get FC weights
fc_weight = net.fc.weight.data  # [1000, 2048]
fc_bias = net.fc.bias.data if net.fc.bias is not None else None

# Get prediction
pred_class = torch.argmax(logits, dim=1).item()
pred_prob = F.softmax(logits, dim=1)[0, pred_class].item()

print(f"Image: {os.path.basename(img_path)}")
print(f"Predicted class: {pred_class}, prob: {pred_prob:.4f}")
print()

# Method 1: My implementation (from visualize_fc_weight_spatial.py)
print("="*60)
print("Method 1: visualize_fc_weight_spatial.py implementation")
print("="*60)

# Features are already after GAP
if len(features.shape) == 4:
    z = features.mean(dim=(2, 3)).squeeze(0)  # Shape: [2048]
else:
    z = features.squeeze(0)  # Already [2048]

# Compute logits and probabilities
logits_check = F.linear(z.unsqueeze(0), fc_weight, fc_bias)
probs_check = F.softmax(logits_check, dim=-1).squeeze(0)

# Compute NLL gradient
grad_logits_my = probs_check.clone()
grad_logits_my[pred_class] -= 1.0

# Compute gradient w.r.t. FC weight for predicted class
grad_W_my = grad_logits_my[pred_class] * z  # [2048]

# Get Fisher matrix for predicted class
fisher_class = fisher_W[pred_class]  # [2048]

# Compute Fisher-weighted gradient
fisher_power = 9.0
fisher_weighted_grad_my = torch.abs(grad_W_my) / (fisher_class ** fisher_power + 1e-10)

# Get top-5 channels
topk_values_my, topk_indices_my = torch.topk(fisher_weighted_grad_my, k=5, largest=True)

print(f"Top-5 channels (my method):")
for i, (idx, val) in enumerate(zip(topk_indices_my, topk_values_my)):
    channel_idx = idx.item()
    weight_val = fc_weight[pred_class, channel_idx].item()
    grad_val = grad_W_my[channel_idx].item()
    fisher_val = fisher_class[channel_idx].item()
    print(f"  Rank {i+1}: channel={channel_idx}, weight={weight_val:.6f}, "
          f"grad={grad_val:.6f}, fisher={fisher_val:.6e}, fisher_grad={val.item():.6e}")
print()

# Method 2: FInD postprocessor method (from find_postprocessor.py)
print("="*60)
print("Method 2: FInD postprocessor implementation")
print("="*60)

# NLL-based gradient: ∇_z L = p - e_y_pred
probs_find = F.softmax(logits, dim=-1).squeeze(0)  # [1000]
grad_logits_find = probs_find.clone()
grad_logits_find[pred_class] -= 1.0

# Compute gradient w.r.t. FC weight
# grad_W = grad_logits @ features (for single sample)
# For single sample: grad_W[c, :] = grad_logits[c] * features[:]
grad_W_find = grad_logits_find[pred_class] * z  # [2048]

# Fisher-weighted gradient
fisher_weighted_grad_find = torch.abs(grad_W_find) / (fisher_class ** fisher_power + 1e-10)

# Get top-5 channels
topk_values_find, topk_indices_find = torch.topk(fisher_weighted_grad_find, k=5, largest=True)

print(f"Top-5 channels (FInD method):")
for i, (idx, val) in enumerate(zip(topk_indices_find, topk_values_find)):
    channel_idx = idx.item()
    weight_val = fc_weight[pred_class, channel_idx].item()
    grad_val = grad_W_find[channel_idx].item()
    fisher_val = fisher_class[channel_idx].item()
    print(f"  Rank {i+1}: channel={channel_idx}, weight={weight_val:.6f}, "
          f"grad={grad_val:.6f}, fisher={fisher_val:.6e}, fisher_grad={val.item():.6e}")
print()

# Compare
print("="*60)
print("Comparison")
print("="*60)

print(f"Channels match: {torch.equal(topk_indices_my, topk_indices_find)}")
print(f"Values match: {torch.allclose(topk_values_my, topk_values_find, rtol=1e-5, atol=1e-8)}")
print(f"Gradients match: {torch.allclose(grad_W_my, grad_W_find, rtol=1e-5, atol=1e-8)}")
print(f"Fisher-weighted gradients match: {torch.allclose(fisher_weighted_grad_my, fisher_weighted_grad_find, rtol=1e-5, atol=1e-8)}")

if torch.equal(topk_indices_my, topk_indices_find):
    print("\n✅ SUCCESS: Both methods produce identical results!")
else:
    print("\n❌ MISMATCH: Methods produce different results")
    print("\nDifferences:")
    for i in range(5):
        if topk_indices_my[i] != topk_indices_find[i]:
            print(f"  Rank {i+1}: my={topk_indices_my[i].item()}, find={topk_indices_find[i].item()}")
