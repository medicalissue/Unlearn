#!/usr/bin/env python
"""
Visualize Spatial Activation of Top-1 Selected FC Weight (FInD-based)
-----------------------------------------------------------------------
Shows which image regions contribute most to the selected FC weight's activation,
using Fisher Information Matrix for selection (same as FInD postprocessor).

For a FC weight W_{c,i}, the spatial contribution is:
    contribution_{h,w} = W_{c,i} * F_{i,h,w}

where F is the feature map before global average pooling.

Selection uses Fisher-weighted gradient: |g_i / F_i^p| where g_i is the NLL gradient
and F_i is the Fisher Information for parameter i.

Usage:
    python scripts/visualize_fc_weight_spatial.py \
        --image-dir temp/ID \
        --checkpoint results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt \
        --arch resnet18 \
        --num-classes 200 \
        --output figures/fc_weight_spatial_activation.pdf
"""

import os
import sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torchvision import transforms as trn
from glob import glob

from openood.networks import ResNet18_224x224, ResNet50


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize FC Weight Spatial Activation')

    parser.add_argument('--image-dir', type=str, default='temp/ID',
                        help='Directory containing images (can specify parent dir to include subdirs)')
    parser.add_argument('--checkpoint', type=str,
                        default='results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt',
                        help='Path to model checkpoint')
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='Network architecture')
    parser.add_argument('--num-classes', type=int, default=200,
                        help='Number of classes')
    parser.add_argument('--output', type=str, default='figures/fc_weight_spatial_activation.pdf',
                        help='Output PDF file path')
    parser.add_argument('--num-images', type=int, default=8,
                        help='Number of images to visualize')
    parser.add_argument('--fisher-cache', type=str, default=None,
                        help='Path to Fisher matrix cache file (if not specified, will auto-detect)')
    parser.add_argument('--fisher-power', type=float, default=1.0,
                        help='Fisher power parameter (p in F^{-p})')

    return parser.parse_args()


def get_model(arch, num_classes, checkpoint_path):
    """Load model with checkpoint or pretrained weights."""
    if arch == 'resnet18':
        net = ResNet18_224x224(num_classes=num_classes)
    elif arch == 'resnet50':
        net = ResNet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Load checkpoint if provided, otherwise use pretrained weights
    if checkpoint_path is None:
        print("No checkpoint provided, using pretrained ImageNet weights...")
        if arch == 'resnet50':
            from torchvision.models import ResNet50_Weights
            from torch.hub import load_state_dict_from_url
            weights = ResNet50_Weights.IMAGENET1K_V1
            net.load_state_dict(load_state_dict_from_url(weights.url))
        else:
            raise ValueError("Pretrained weights only available for ResNet50")
    else:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        net.load_state_dict(ckpt)

    net.eval()
    return net


class FeatureExtractor(nn.Module):
    """Extract features before GAP for spatial analysis."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = None

        # Register hook to capture features before GAP
        # For ResNet, this is after layer4 (before avgpool)
        if hasattr(model, 'layer4'):
            model.layer4.register_forward_hook(self._hook_fn)
        else:
            raise ValueError("Model architecture not supported")

    def _hook_fn(self, module, input, output):
        """Hook to capture feature maps."""
        self.features = output

    def forward(self, x):
        """Forward pass and return both logits and features."""
        logits = self.model(x)
        features = self.features  # Shape: [B, C, H, W]
        return logits, features


def load_fisher_matrix(fisher_cache_path):
    """Load Fisher Information Matrix from cache file.

    Returns:
        fisher_W: Fisher matrix for FC weights, shape [num_classes, feature_dim]
    """
    if not os.path.exists(fisher_cache_path):
        raise FileNotFoundError(f"Fisher cache not found: {fisher_cache_path}")

    print(f"Loading Fisher matrix from {fisher_cache_path}")
    cache = torch.load(fisher_cache_path, map_location='cpu')
    fisher_W = cache['fisher_W']
    print(f"  Fisher matrix shape: {fisher_W.shape}")
    print(f"  Dataset: {cache.get('dataset_name', 'unknown')}")
    print(f"  Model: {cache.get('model_arch', 'unknown')}")

    return fisher_W


def find_fisher_cache(checkpoint_path, arch, num_classes, dataset='imagenet200'):
    """Auto-detect Fisher cache path based on checkpoint."""
    # Try common cache locations
    ckpt_dir = os.path.dirname(checkpoint_path)
    ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]

    # Pattern: fisher_cache_{dataset}_{arch}_{ckpt}_nll.pt
    cache_name = f"fisher_cache_{dataset}_{arch}_{ckpt_name}_nll.pt"

    # Search in checkpoint directory and parent directories
    search_dirs = [
        ckpt_dir,
        os.path.join(ckpt_dir, '..'),
        os.path.join(ROOT_DIR, 'results', 'fisher_cache'),
    ]

    for search_dir in search_dirs:
        cache_path = os.path.join(search_dir, cache_name)
        if os.path.exists(cache_path):
            return cache_path

    # If not found, return the expected path in checkpoint directory
    return os.path.join(ckpt_dir, cache_name)


def compute_topk_spatial_activation(features, fc_weight, fc_bias, predicted_class, fisher_W, fisher_power=1.0, rank=1, use_min=False):
    """
    Compute spatial activation for top-k selected FC weight using Fisher.

    Args:
        features: Feature maps before GAP, shape [B, C, H, W]
        fc_weight: FC layer weight matrix, shape [num_classes, C]
        fc_bias: FC layer bias, shape [num_classes] or None
        predicted_class: Predicted class index
        fisher_W: Fisher matrix for FC weights, shape [num_classes, C]
        fisher_power: Fisher power parameter (p in F^{-p})
        rank: Which top-k channel to select (1=top-1, 2=top-2, etc.)
        use_min: If True, select from bottom-k channels instead

    Returns:
        selected_channel: Channel index at specified rank
        spatial_activation: Spatial heatmap, shape [H, W]
        weight_value: The FC weight value W_{c,i}
        gradient_value: The NLL gradient value
        fisher_weighted_grad_value: The Fisher-weighted gradient value for selected channel
        fisher_weighted_grad_all: All Fisher-weighted gradient values, shape [C]
    """
    B, C, H, W = features.shape
    assert B == 1, "Only single image supported"

    # Get FC weights for predicted class
    class_weights = fc_weight[predicted_class]  # Shape: [C]

    # Global Average Pooling
    z = features.mean(dim=(2, 3)).squeeze(0)  # Shape: [C]

    # Compute logits and probabilities
    logits = F.linear(z.unsqueeze(0), fc_weight, fc_bias)  # [1, num_classes]
    probs = F.softmax(logits, dim=-1).squeeze(0)  # [num_classes]

    # Compute NLL gradient: ∇_z L = p - e_y
    grad_logits = probs.clone()  # [num_classes]
    grad_logits[predicted_class] -= 1.0

    # Compute gradient w.r.t. FC weight for predicted class: grad_W[c, :] = grad_logits[c] * z
    grad_W = grad_logits[predicted_class] * z  # [C]

    # Get Fisher matrix for predicted class
    fisher_class = fisher_W[predicted_class]  # [C]

    # Compute Fisher-weighted gradient: |g_i / F_i^p|
    # Add epsilon for numerical stability
    fisher_weighted_grad = torch.abs(grad_W) / (fisher_class ** fisher_power + 1e-10)

    # Get top-k channels and select the one at specified rank
    if use_min:
        # Get smallest values
        topk_values, topk_indices = torch.topk(fisher_weighted_grad, k=min(rank, C), largest=False)
    else:
        # Get largest values
        topk_values, topk_indices = torch.topk(fisher_weighted_grad, k=min(rank, C), largest=True)

    # Select channel at specified rank (rank-1 because 0-indexed)
    selected_channel = topk_indices[rank - 1].item()

    weight_value = class_weights[selected_channel].item()
    gradient_value = grad_W[selected_channel].item()
    fisher_weighted_grad_value = fisher_weighted_grad[selected_channel].item()

    # Compute spatial activation for this channel
    # contribution_{h,w} = W_{c,i} * F_{i,h,w}
    channel_feature = features[0, selected_channel].cpu().numpy()  # Shape: [H, W]
    spatial_activation = weight_value * channel_feature

    return selected_channel, spatial_activation, weight_value, gradient_value, fisher_weighted_grad_value, fisher_weighted_grad


def visualize_spatial_activation(image_path, model, preprocessor, fisher_W, fisher_power=1.0, output_path=None):
    """
    Visualize spatial activation for a single image using Fisher-weighted selection.

    Returns:
        fig: matplotlib figure
        info: dict with metadata
    """
    # Load and preprocess image
    img_pil = Image.open(image_path).convert('RGB')
    img_tensor = preprocessor(img_pil).unsqueeze(0)  # [1, 3, 224, 224]

    # Create feature extractor
    feature_extractor = FeatureExtractor(model)

    # Forward pass
    with torch.no_grad():
        logits, features = feature_extractor(img_tensor)

    # Get prediction
    pred_class = torch.argmax(logits, dim=1).item()
    pred_prob = F.softmax(logits, dim=1)[0, pred_class].item()

    # Get FC layer weights and bias
    if hasattr(model, 'fc'):
        fc_weight = model.fc.weight.data  # Shape: [num_classes, C]
        fc_bias = model.fc.bias.data if model.fc.bias is not None else None
    elif hasattr(model, 'linear'):
        fc_weight = model.linear.weight.data
        fc_bias = model.linear.bias.data if model.linear.bias is not None else None
    else:
        raise ValueError("Cannot find FC layer")

    # Compute spatial activations for top-1, top-2, and top-5 channels
    results = []
    for rank in [1, 2, 5]:
        channel, spatial_act, weight_val, grad_val, fisher_grad_val, fisher_grad_all = compute_topk_spatial_activation(
            features, fc_weight, fc_bias, pred_class, fisher_W, fisher_power, rank=rank, use_min=False
        )
        results.append({
            'rank': rank,
            'channel': channel,
            'spatial_activation': spatial_act,
            'weight_value': weight_val,
            'gradient_value': grad_val,
            'fisher_weighted_grad_value': fisher_grad_val,
            'fisher_weighted_grad_all': fisher_grad_all
        })

    # Resize original image to match feature map size
    H, W = results[0]['spatial_activation'].shape

    # Center crop image to square
    width, height = img_pil.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    img_square = img_pil.crop((left, top, left + size, top + size))

    # Create visualization with 3 rows (top-1, top-2, top-5) × 4 columns (original, heatmap, overlay, distribution)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    filename = os.path.basename(image_path)

    # Import zoom for later use
    from scipy.ndimage import zoom

    # Loop through each rank (row)
    for row_idx, result in enumerate(results):
        rank = result['rank']
        channel = result['channel']
        spatial_activation = result['spatial_activation']
        weight_value = result['weight_value']
        fisher_weighted_grad_all = result['fisher_weighted_grad_all']

        # Column 0: Original image (only in first row with title)
        if row_idx == 0:
            axes[row_idx, 0].imshow(img_square)
            axes[row_idx, 0].set_title(f'{filename}\nClass: {pred_class}, Prob: {pred_prob:.3f}', fontsize=12)
        else:
            axes[row_idx, 0].imshow(img_square)
            axes[row_idx, 0].set_title(f'Top-{rank}', fontsize=12, fontweight='bold')
        axes[row_idx, 0].axis('off')
        axes[row_idx, 0].set_aspect('equal')

        # Determine colormap based on weight sign
        if weight_value < 0:
            heatmap_cmap = 'RdBu'  # Inverted: blue for positive
            overlay_cmap = 'RdBu'
        else:
            heatmap_cmap = 'RdBu_r'  # Normal: red for positive
            overlay_cmap = 'RdBu_r'

        # Column 1: Spatial activation heatmap
        im1 = axes[row_idx, 1].imshow(spatial_activation, cmap=heatmap_cmap, interpolation='bilinear')
        axes[row_idx, 1].set_title(f'Spatial Activation (Top-{rank})\nChannel: {channel}, Weight: {weight_value:.4f}',
                         fontsize=11)
        axes[row_idx, 1].axis('off')
        axes[row_idx, 1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # Column 2: Overlay on original image
        axes[row_idx, 2].imshow(img_square)

        # Normalize activation to [0, 1] for overlay
        activation_norm = spatial_activation.copy()
        abs_max = max(abs(activation_norm.min()), abs(activation_norm.max()))
        if abs_max > 1e-8:
            activation_norm = activation_norm / abs_max  # [-1, 1]
            activation_norm = (activation_norm + 1) / 2  # [0, 1]
        else:
            activation_norm = np.ones_like(activation_norm) * 0.5

        # Resize to square image size
        zoom_factor = (size / H, size / W)
        activation_resized = zoom(activation_norm, zoom_factor, order=1)

        # Apply overlay
        im2 = axes[row_idx, 2].imshow(activation_resized, cmap=overlay_cmap, alpha=0.5, interpolation='bilinear', vmin=0, vmax=1)
        if weight_value < 0:
            axes[row_idx, 2].set_title(f'Overlay (Alpha=0.5)\nBlue=Pos, Red=Neg', fontsize=11)
        else:
            axes[row_idx, 2].set_title(f'Overlay (Alpha=0.5)\nRed=Pos, Blue=Neg', fontsize=11)
        axes[row_idx, 2].axis('off')
        axes[row_idx, 2].set_aspect('equal')

        # Column 3: Fisher-weighted gradient distribution
        fisher_grad_np = fisher_weighted_grad_all.cpu().numpy()
        top_k = 20
        top_indices = np.argsort(fisher_grad_np)[-top_k:][::-1]
        top_values = fisher_grad_np[top_indices]

        # Create bar plot with selected channel highlighted
        colors = ['red' if i == channel else 'gray' for i in top_indices]
        axes[row_idx, 3].bar(range(top_k), top_values, color=colors, alpha=0.7)
        axes[row_idx, 3].set_xlabel('Channel Rank', fontsize=10)
        axes[row_idx, 3].set_ylabel('Fisher-weighted Gradient', fontsize=10)
        axes[row_idx, 3].set_title(f'Top-{top_k} Channels (Red=Top-{rank})', fontsize=11)
        axes[row_idx, 3].set_yscale('log')
        axes[row_idx, 3].grid(True, alpha=0.3, axis='y')

        # Add value annotation for selected channel
        selected_rank_in_top20 = np.where(top_indices == channel)[0]
        if len(selected_rank_in_top20) > 0:
            selected_rank_in_top20 = selected_rank_in_top20[0]
            axes[row_idx, 3].text(selected_rank_in_top20, top_values[selected_rank_in_top20],
                        f'{top_values[selected_rank_in_top20]:.1f}',
                        ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    info = {
        'pred_class': pred_class,
        'pred_prob': pred_prob,
        'results': results  # Store all results for top-1, top-2, top-5
    }

    return fig, info


def main():
    args = parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = get_model(args.arch, args.num_classes, args.checkpoint)
    model.eval()

    # Load Fisher matrix
    if args.fisher_cache is not None:
        fisher_cache_path = args.fisher_cache
    else:
        # Auto-detect Fisher cache
        fisher_cache_path = find_fisher_cache(args.checkpoint, args.arch, args.num_classes)

    fisher_W = load_fisher_matrix(fisher_cache_path)
    print(f"Fisher power: {args.fisher_power}")

    # Preprocessor
    preprocessor = trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225])
    ])

    # Get ID images: Random cat images from ImageNet-1K validation
    # Cat classes in ImageNet-1K: 281-285 (tabby, tiger cat, Persian cat, Siamese cat, Egyptian cat)
    cat_classes = list(range(281, 286))  # [281, 282, 283, 284, 285]

    print("Collecting ID images (cats from ImageNet-1K val)...")
    id_image_paths = []
    val_dir = 'data/images_largescale/imagenet_1k/val'

    # Load validation labels to filter cat images
    import json
    val_labels_file = 'data/benchmark_imglist/imagenet/test_imagenet.txt'
    if os.path.exists(val_labels_file):
        with open(val_labels_file, 'r') as f:
            val_data = [line.strip().split() for line in f if line.strip()]
            for img_path, label in val_data:
                if int(label) in cat_classes:
                    full_path = os.path.join('data/images_largescale', img_path)
                    if os.path.exists(full_path):
                        id_image_paths.append(full_path)

    # If no labeled file, just get first few val images as fallback
    if len(id_image_paths) == 0:
        print("  Warning: Could not find labeled validation file, using first images from val")
        id_image_paths = sorted(glob(os.path.join(val_dir, '*.JPEG')))[:args.num_images]
    else:
        # Randomly sample cat images
        import random
        random.seed(42)
        id_image_paths = random.sample(id_image_paths, min(args.num_images, len(id_image_paths)))

    print(f"  Found {len(id_image_paths)} ID images (cats)")

    # Get OOD images from temp/OOD folder
    print("Collecting OOD images (from temp/OOD folder)...")
    ood_image_paths = []
    for ext in ['*.JPEG', '*.jpg', '*.png', '*.JPG', '*.PNG']:
        ood_image_paths.extend(glob(f'temp/OOD/{ext}', recursive=False))
    ood_image_paths = sorted(set(ood_image_paths))  # Remove duplicates and sort
    if len(ood_image_paths) == 0:
        print("  Warning: No OOD images found in temp/OOD folder")
    print(f"  Found {len(ood_image_paths)} OOD images")

    # Combine ID and OOD images
    image_paths = id_image_paths + ood_image_paths
    image_labels = ['ID (Cat)'] * len(id_image_paths) + ['OOD'] * len(ood_image_paths)

    print(f"\nTotal: {len(image_paths)} images ({len(id_image_paths)} ID + {len(ood_image_paths)} OOD)")

    # Create multi-page visualization
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with PdfPages(args.output) as pdf:
        for i, (img_path, label) in enumerate(zip(image_paths, image_labels)):
            print(f"\n[{i+1}/{len(image_paths)}] [{label}] Processing {os.path.basename(img_path)}...")

            # Visualize
            fig, info = visualize_spatial_activation(img_path, model, preprocessor, fisher_W, args.fisher_power)

            # Update title to include ID/OOD label (prepend to existing title)
            filename = os.path.basename(img_path)
            fig.axes[0].set_title(f'[{label}] {filename}\nClass: {info["pred_class"]}, Prob: {info["pred_prob"]:.3f}', fontsize=12)

            # Print info
            print(f"  Predicted class: {info['pred_class']} (prob: {info['pred_prob']:.3f})")
            print(f"  Top-1 channel: {info['top1_channel']}, weight: {info['weight_value']:.4f}")
            print(f"  Gradient: {info['gradient_value']:.4f}, Fisher-weighted: {info['fisher_weighted_grad']:.6f}")
            print(f"  Activation range: [{info['activation_min']:.4f}, {info['activation_max']:.4f}]")

            # Save to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"\n✓ Saved visualization to {args.output}")


if __name__ == '__main__':
    main()
