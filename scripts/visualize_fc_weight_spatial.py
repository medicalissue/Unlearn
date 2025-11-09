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
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of top-k parameters to average for spatial activation')

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


def compute_topk_spatial_activation(features, fc_weight, fc_bias, predicted_class, fisher_W, fisher_power=1.0, rank=1, use_min=True):
    """
    Compute spatial activation for top-k selected FC weight using Fisher.

    This follows FInD postprocessor logic:
    1. Compute gradients for ALL classes (not just predicted)
    2. Flatten all parameters: [num_classes * feature_dim]
    3. Compute Fisher-weighted gradient for all parameters
    4. Select top-k parameters globally

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
        selected_class: Class index of selected parameter
        selected_channel: Channel index of selected parameter
        spatial_activation: Spatial heatmap, shape [H, W]
        weight_value: The FC weight value W_{c,i}
        gradient_value: The NLL gradient value
        fisher_weighted_grad_value: The Fisher-weighted gradient value for selected parameter
        fisher_weighted_grad_all: All Fisher-weighted gradient values, shape [num_classes * C]
    """
    B, C, H, W = features.shape
    assert B == 1, "Only single image supported"
    num_classes = fc_weight.shape[0]

    # Global Average Pooling
    z = features.mean(dim=(2, 3)).squeeze(0)  # Shape: [C]

    # Compute logits and probabilities
    logits = F.linear(z.unsqueeze(0), fc_weight, fc_bias)  # [1, num_classes]
    probs = F.softmax(logits, dim=-1).squeeze(0)  # [num_classes]

    # Compute NLL gradient: ∇_z L = p - e_y_pred
    grad_logits = probs.clone()  # [num_classes]
    grad_logits[predicted_class] -= 1.0

    # Compute gradient w.r.t. FC weight for ALL classes
    # grad_W[c, i] = grad_logits[c] * z[i]
    grad_W = torch.einsum('c,d->cd', grad_logits, z)  # [num_classes, C]

    # Flatten gradients and Fisher matrix
    grad_W_flat = grad_W.flatten()  # [num_classes * C]
    fisher_W_flat = fisher_W.flatten()  # [num_classes * C]

    # Compute Fisher-weighted gradient: |g / F^p| for ALL parameters
    fisher_weighted_grad = torch.abs(grad_W_flat) / (fisher_W_flat ** fisher_power + 1e-10)

    # Get top-k parameters and select the one at specified rank
    if use_min:
        topk_values, topk_indices = torch.topk(fisher_weighted_grad, k=min(rank, len(fisher_weighted_grad)), largest=False)
    else:
        topk_values, topk_indices = torch.topk(fisher_weighted_grad, k=min(rank, len(fisher_weighted_grad)), largest=True)

    # Select parameter at specified rank (rank-1 because 0-indexed)
    selected_idx = topk_indices[rank - 1].item()

    # Convert flat index to (class, channel)
    selected_class = selected_idx // C
    selected_channel = selected_idx % C

    weight_value = fc_weight[selected_class, selected_channel].item()
    gradient_value = grad_W[selected_class, selected_channel].item()
    fisher_weighted_grad_value = fisher_weighted_grad[selected_idx].item()

    # Compute spatial activation for this channel
    # contribution_{h,w} = W_{c,i} * F_{i,h,w}
    channel_feature = features[0, selected_channel].cpu().numpy()  # Shape: [H, W]
    spatial_activation = weight_value * channel_feature

    return selected_class, selected_channel, spatial_activation, weight_value, gradient_value, fisher_weighted_grad_value, fisher_weighted_grad


def compute_topk_avg_spatial_activation(features, fc_weight, fc_bias, predicted_class, fisher_W, fisher_power=1.0, k=10):
    """
    Compute averaged spatial activation for top-k selected FC weights using Fisher.

    Similar to GradCAM, this averages spatial activations from multiple top-k parameters.

    Args:
        features: Feature maps before GAP, shape [B, C, H, W]
        fc_weight: FC layer weight matrix, shape [num_classes, feature_dim]
        fc_bias: FC layer bias, shape [num_classes] or None
        predicted_class: Predicted class index
        fisher_W: Fisher matrix for FC weights, shape [num_classes, feature_dim]
        fisher_power: Fisher power parameter (p in F^{-p})
        k: Number of top parameters to average (default: 10)

    Returns:
        spatial_activation_avg: Averaged spatial heatmap, shape [H, W]
        topk_info: List of dicts with info for each top-k parameter
        fisher_weighted_grad_all: All Fisher-weighted gradient values
    """
    B, C, H, W = features.shape
    assert B == 1, "Only single image supported"
    num_classes = fc_weight.shape[0]

    # Global Average Pooling
    z = features.mean(dim=(2, 3)).squeeze(0)  # Shape: [C]

    # Compute logits and probabilities
    logits = F.linear(z.unsqueeze(0), fc_weight, fc_bias)  # [1, num_classes]
    probs = F.softmax(logits, dim=-1).squeeze(0)  # [num_classes]

    # Compute NLL gradient: ∇_z L = p - e_y_pred
    grad_logits = probs.clone()  # [num_classes]
    grad_logits[predicted_class] -= 1.0

    # Compute gradient w.r.t. FC weight for ALL classes
    grad_W = torch.einsum('c,d->cd', grad_logits, z)  # [num_classes, C]

    # Flatten gradients and Fisher matrix
    grad_W_flat = grad_W.flatten()  # [num_classes * C]
    fisher_W_flat = fisher_W.flatten()  # [num_classes * C]

    # Compute Fisher-weighted gradient: |g / F^p| for ALL parameters
    fisher_weighted_grad = torch.abs(grad_W_flat) / (fisher_W_flat ** fisher_power + 1e-10)

    # Get top-k parameters
    topk_values, topk_indices = torch.topk(fisher_weighted_grad, k=min(k, len(fisher_weighted_grad)), largest=True)

    # Compute spatial activation for each top-k parameter and average
    spatial_activations = []
    topk_info = []

    for i, (idx, fisher_grad_val) in enumerate(zip(topk_indices, topk_values)):
        idx = idx.item()

        # Convert flat index to (class, channel)
        param_class = idx // C
        param_channel = idx % C

        weight_val = fc_weight[param_class, param_channel].item()
        grad_val = grad_W[param_class, param_channel].item()

        # Compute spatial activation: W_{c,i} * F_{i,h,w}
        channel_feature = features[0, param_channel].cpu().numpy()  # Shape: [H, W]
        spatial_act = weight_val * channel_feature

        spatial_activations.append(spatial_act)
        topk_info.append({
            'rank': i + 1,
            'class': param_class,
            'channel': param_channel,
            'weight': weight_val,
            'gradient': grad_val,
            'fisher_grad': fisher_grad_val.item()
        })

    # Average spatial activations
    spatial_activation_avg = np.mean(spatial_activations, axis=0)  # Shape: [H, W]

    return spatial_activation_avg, topk_info, fisher_weighted_grad


def visualize_spatial_activation(image_path, model, preprocessor, fisher_W, fisher_power=1.0, topk=10, output_path=None):
    """
    Visualize spatial activation for a single image using Fisher-weighted selection.

    Args:
        image_path: Path to image file
        model: Neural network model
        preprocessor: Image preprocessing pipeline
        fisher_W: Fisher Information Matrix
        fisher_power: Fisher power parameter
        topk: Number of top-k parameters to average
        output_path: Optional path to save visualization

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

    # Compute averaged spatial activation for top-k parameters (GradCAM-style)
    spatial_activation, topk_info, fisher_weighted_grad_all = compute_topk_avg_spatial_activation(
        features, fc_weight, fc_bias, pred_class, fisher_W, fisher_power, k=topk
    )

    # Resize original image to match feature map size
    H, W = spatial_activation.shape

    # Center crop image to square
    width, height = img_pil.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    img_square = img_pil.crop((left, top, left + size, top + size))

    # Create visualization with 1 row × 4 columns (original, heatmap, overlay, distribution)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    filename = os.path.basename(image_path)

    # Import zoom for later use
    from scipy.ndimage import zoom

    # Column 0: Original image
    axes[0].imshow(img_square)
    axes[0].set_title(f'{filename}\nClass: {pred_class}, Prob: {pred_prob:.3f}', fontsize=12)
    axes[0].axis('off')
    axes[0].set_aspect('equal')

    # Use RdBu_r for heatmap to show both positive and negative
    heatmap_cmap = 'RdBu_r'

    # Column 1: Spatial activation heatmap (averaged over top-k)
    im1 = axes[1].imshow(spatial_activation, cmap=heatmap_cmap, interpolation='bilinear')
    axes[1].set_title(f'Spatial Activation (Top-{topk} Avg)\nFisher Power: {fisher_power:.1f}',
                     fontsize=11)
    axes[1].axis('off')
    axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Column 2: Enhanced GradCAM-style overlay
    axes[2].imshow(img_square)

    # Normalize spatial activation to [0, 1] range
    activation_gradcam = spatial_activation.copy()
    activation_min = activation_gradcam.min()
    activation_max = activation_gradcam.max()
    activation_range = activation_max - activation_min

    if activation_range > 0:
        # Normalize to [0, 1]
        activation_gradcam = (activation_gradcam - activation_min) / activation_range
    else:
        activation_gradcam = np.zeros_like(activation_gradcam)

    # Resize to match image size with smooth interpolation
    zoom_factor = (size / H, size / W)
    activation_resized = zoom(activation_gradcam, zoom_factor, order=3)  # Cubic interpolation for smoothness

    # Apply Gaussian smoothing for better visual quality
    from scipy.ndimage import gaussian_filter
    activation_smooth = gaussian_filter(activation_resized, sigma=2.0)

    # Enhance contrast using histogram equalization-like transform
    # Apply power transformation to make important regions stand out
    activation_enhanced = np.clip(activation_smooth, 0, 1) ** 0.7  # Lower power = more contrast

    # Create RGBA image with YlOrRd colormap (Yellow -> Orange -> Red)
    # Low values: transparent, High values: Yellow -> Orange -> Red
    import matplotlib.cm as cm
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('YlOrRd')
    rgba = cmap(activation_enhanced)

    # Smooth alpha channel for natural blending
    # Use activation value directly for alpha (higher activation = more visible)
    alpha_base = 0.7
    alpha_channel = alpha_base * activation_smooth  # Direct proportional to activation
    rgba[..., 3] = alpha_channel

    im2 = axes[2].imshow(rgba, interpolation='bilinear')
    axes[2].set_title(f'Enhanced Overlay (Top-{topk})\nYellow→Orange→Red', fontsize=11)
    axes[2].axis('off')
    axes[2].set_aspect('equal')

    # Column 3: Top-k parameter info
    # Show Fisher-weighted gradients for the top-k parameters used in averaging
    fisher_grads_topk = [info['fisher_grad'] for info in topk_info]

    # Create bar plot highlighting top-k parameters
    axes[3].bar(range(len(topk_info)), fisher_grads_topk, color='darkgreen', alpha=0.7)
    axes[3].set_xlabel('Parameter Rank', fontsize=10)
    axes[3].set_ylabel('Fisher-weighted Gradient', fontsize=10)
    axes[3].set_title(f'Top-{topk} Parameters (Averaged)', fontsize=11)
    axes[3].set_yscale('log')
    axes[3].grid(True, alpha=0.3, axis='y')

    # Add annotations for top-3
    for i in range(min(3, len(topk_info))):
        axes[3].text(i, fisher_grads_topk[i],
                    f'{fisher_grads_topk[i]:.1e}',
                    ha='center', va='bottom', fontsize=7, color='darkgreen', fontweight='bold')

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    info = {
        'pred_class': pred_class,
        'pred_prob': pred_prob,
        'topk_info': topk_info,
        'spatial_activation': spatial_activation
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

    # Get ID images from temp/ID folder
    print("Collecting ID images (from temp/ID folder)...")
    id_image_paths = []
    for ext in ['*.JPEG', '*.jpg', '*.png', '*.JPG', '*.PNG']:
        id_image_paths.extend(glob(f'temp/ID/{ext}', recursive=False))
    id_image_paths = sorted(set(id_image_paths))  # Remove duplicates and sort
    if len(id_image_paths) == 0:
        print("  Warning: No ID images found in temp/ID folder")
    print(f"  Found {len(id_image_paths)} ID images")

    # Get OOD images from temp folder (root level)
    print("Collecting OOD images (from temp/ folder)...")
    ood_image_paths = []
    for ext in ['*.JPEG', '*.jpg', '*.png', '*.JPG', '*.PNG', '*.jpeg']:
        ood_image_paths.extend(glob(f'temp/{ext}', recursive=False))
    ood_image_paths = sorted(set(ood_image_paths))  # Remove duplicates and sort
    if len(ood_image_paths) == 0:
        print("  Warning: No OOD images found in temp/ folder")
    print(f"  Found {len(ood_image_paths)} OOD images")

    # Combine ID and OOD images
    image_paths = id_image_paths + ood_image_paths
    image_labels = ['ID (Pizza)'] * len(id_image_paths) + ['OOD (Donuts)'] * len(ood_image_paths)

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
            fig, info = visualize_spatial_activation(img_path, model, preprocessor, fisher_W, args.fisher_power, args.topk)

            # Update title to include ID/OOD label (prepend to existing title)
            filename = os.path.basename(img_path)
            fig.axes[0].set_title(f'[{label}] {filename}\nClass: {info["pred_class"]}, Prob: {info["pred_prob"]:.3f}', fontsize=12)

            # Print info
            print(f"  Predicted class: {info['pred_class']} (prob: {info['pred_prob']:.3f})")
            print(f"  Top-{args.topk} params averaged for spatial activation:")
            for top_info in info['topk_info'][:3]:  # Show top-3
                print(f"    Rank {top_info['rank']}: class={top_info['class']}, channel={top_info['channel']}, "
                      f"weight={top_info['weight']:.4f}, fisher_grad={top_info['fisher_grad']:.6e}")
            print(f"  Spatial activation range: [{info['spatial_activation'].min():.4f}, {info['spatial_activation'].max():.4f}]")

            # Save to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"\n✓ Saved visualization to {args.output}")


if __name__ == '__main__':
    main()
