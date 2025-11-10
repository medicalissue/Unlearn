#!/usr/bin/env python
"""
Create Figure 1 style visualization showing ID and OOD images with overlays.
Layout: 2 rows × 4 columns
Row 1: ID original, ID overlay, OOD original, OOD overlay
Row 2: ID original, ID overlay, OOD original, OOD overlay
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as trn
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
import matplotlib

# Load model
from openood.networks import ResNet50

def get_model(arch='resnet50', num_classes=1000):
    """Load ResNet50 with ImageNet weights."""
    net = ResNet50(num_classes=num_classes)
    ckpt = torch.load('results/pretrained_weights/resnet50_imagenet1k_v1.pth', map_location='cpu')
    net.load_state_dict(ckpt)
    net.eval()
    return net


class FeatureExtractor(nn.Module):
    """Extract features before GAP for spatial analysis."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = None
        self._register_hooks()

    def _register_hooks(self):
        def hook(module, input, output):
            self.features = output

        # ResNet50: layer4 output before avgpool
        if hasattr(self.model, 'layer4'):
            self.model.layer4.register_forward_hook(hook)
        else:
            raise ValueError("Cannot find layer4 in model")

    def forward(self, x):
        logits = self.model(x)
        features = self.features
        return logits, features


def load_fisher_matrix(fisher_cache_path):
    """Load Fisher Information Matrix from cache file."""
    cache = torch.load(fisher_cache_path, map_location='cpu')
    fisher_W = cache['fisher_W']
    return fisher_W


def compute_nll_gradcam(features, fc_weight, fc_bias, predicted_class):
    """
    Compute Grad-CAM using NLL (Negative Log-Likelihood) as the target.

    Returns spatial activation map based on NLL gradient.
    """
    B, C, H, W = features.shape
    assert B == 1, "Only single image supported"
    num_classes = fc_weight.shape[0]

    # Enable gradient computation for features
    features_grad = features.clone().detach().requires_grad_(True)

    # Global Average Pooling
    z = features_grad.mean(dim=(2, 3)).squeeze(0)  # Shape: [C]

    # Compute logits and probabilities
    logits = F.linear(z.unsqueeze(0), fc_weight, fc_bias)  # [1, num_classes]
    probs = F.softmax(logits, dim=-1).squeeze(0)  # [num_classes]

    # NLL: -log(p_pred)
    nll = -torch.log(probs[predicted_class] + 1e-10)

    # Backprop to get gradients w.r.t. features
    nll.backward()
    feature_grads = features_grad.grad  # Shape: [1, C, H, W]

    # Grad-CAM: ReLU(weighted combination of feature maps)
    weights = feature_grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    gradcam = (weights * features).sum(dim=1).squeeze(0)  # [H, W]
    gradcam = F.relu(gradcam).detach().cpu().numpy()

    return gradcam


def compute_kl_uniform_gradcam(features, fc_weight, fc_bias, predicted_class):
    """
    Compute Grad-CAM using KL divergence over uniform distribution as the target.

    Returns spatial activation map based on KL(pred || uniform) gradient.
    """
    B, C, H, W = features.shape
    assert B == 1, "Only single image supported"
    num_classes = fc_weight.shape[0]

    # Enable gradient computation for features
    features_grad = features.clone().detach().requires_grad_(True)

    # Global Average Pooling
    z = features_grad.mean(dim=(2, 3)).squeeze(0)  # Shape: [C]

    # Compute logits and probabilities
    logits = F.linear(z.unsqueeze(0), fc_weight, fc_bias)  # [1, num_classes]
    probs = F.softmax(logits, dim=-1).squeeze(0)  # [num_classes]

    # KL divergence: KL(p || uniform) = sum(p * log(p * num_classes))
    uniform_prob = 1.0 / num_classes
    kl_div = (probs * (torch.log(probs + 1e-10) - torch.log(torch.tensor(uniform_prob)))).sum()

    # Backprop to get gradients w.r.t. features
    kl_div.backward()
    feature_grads = features_grad.grad  # Shape: [1, C, H, W]

    # Grad-CAM: ReLU(weighted combination of feature maps)
    weights = feature_grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    gradcam = (weights * features).sum(dim=1).squeeze(0)  # [H, W]
    gradcam = F.relu(gradcam).detach().cpu().numpy()

    return gradcam


def compute_gtg_topk_gradcam(features, fc_weight, fc_bias, predicted_class, k=10):
    """
    Compute Grad-CAM using top-k parameters selected by g^T g (gradient magnitude squared).

    Args:
        k: Number of top parameters to use. If k=-1, use all parameters.

    Returns averaged spatial activation map from top-k parameters.
    """
    B, C, H, W = features.shape
    assert B == 1, "Only single image supported"
    num_classes = fc_weight.shape[0]

    # Enable gradient computation for features
    features_grad = features.clone().detach().requires_grad_(True)

    # Global Average Pooling
    z = features_grad.mean(dim=(2, 3)).squeeze(0)  # Shape: [C]

    # Compute logits and probabilities
    logits = F.linear(z.unsqueeze(0), fc_weight, fc_bias)  # [1, num_classes]
    probs = F.softmax(logits, dim=-1).squeeze(0)  # [num_classes]

    # Compute NLL gradient: ∇_z L = p - e_y_pred
    grad_logits = probs.clone()  # [num_classes]
    grad_logits[predicted_class] -= 1.0

    # Compute gradient w.r.t. FC weight for ALL classes
    grad_W = torch.einsum('c,d->cd', grad_logits, z)  # [num_classes, C]

    # Flatten gradients
    grad_W_flat = grad_W.flatten()  # [num_classes * C]

    # Compute g^T g (gradient magnitude squared)
    gtg = grad_W_flat ** 2

    # Get top-k parameters (or all if k=-1)
    if k == -1:
        topk_indices = torch.arange(len(gtg))
    else:
        topk_values, topk_indices = torch.topk(gtg, k=min(k, len(gtg)), largest=True)

    # Compute gradient w.r.t. feature maps for Grad-CAM
    target_logit = logits[0, predicted_class]
    target_logit.backward(retain_graph=True)
    feature_grads = features_grad.grad  # Shape: [1, C, H, W]

    # Compute Grad-CAM spatial contributions for each top-k parameter and average
    spatial_contributions = []

    for idx in topk_indices:
        idx = idx.item()

        # Convert flat index to (class, channel)
        param_class = idx // C
        param_channel = idx % C

        # Get spatial feature map for this channel
        channel_feature = features[0, param_channel].detach().cpu().numpy()  # Shape: [H, W]

        # Get Grad-CAM weight: global average pooling of gradients for this channel
        grad_cam_weight = feature_grads[0, param_channel].mean().item()

        # Compute Grad-CAM spatial contribution
        gradcam_contribution = grad_cam_weight * channel_feature

        spatial_contributions.append(gradcam_contribution)

    # Average Grad-CAM spatial contributions
    spatial_contribution_avg = np.mean(spatial_contributions, axis=0)  # Shape: [H, W]

    return spatial_contribution_avg


def compute_fisher_weighted_gradcam(features, fc_weight, fc_bias, predicted_class, fisher_W, fisher_power=1.0, k=10):
    """
    Compute averaged Fisher-weighted Grad-CAM spatial contributions for top-k selected FC weights.

    Args:
        k: Number of top parameters to use. If k=-1, use all parameters.

    Uses g^2 / F^p (gradient squared over Fisher) for parameter selection.
    Returns Fisher-weighted Grad-CAM contribution: (g^2 / F^p) * (∂y/∂F_i) * F_{i,h,w}
    """
    B, C, H, W = features.shape
    assert B == 1, "Only single image supported"
    num_classes = fc_weight.shape[0]

    # Enable gradient computation for features
    features_grad = features.clone().detach().requires_grad_(True)

    # Global Average Pooling
    z = features_grad.mean(dim=(2, 3)).squeeze(0)  # Shape: [C]

    # Compute logits and probabilities
    logits = F.linear(z.unsqueeze(0), fc_weight, fc_bias)  # [1, num_classes]
    probs = F.softmax(logits, dim=-1).squeeze(0)  # [num_classes]

    # Compute NLL gradient: ∇_z L = p - e_y_pred
    grad_logits = probs.clone()  # [num_classes]
    grad_logits[predicted_class] -= 1.0

    # Compute gradient w.r.t. FC weight for ALL classes (for Fisher weighting)
    grad_W = torch.einsum('c,d->cd', grad_logits, z)  # [num_classes, C]

    # Flatten gradients and Fisher matrix
    grad_W_flat = grad_W.flatten()  # [num_classes * C]
    fisher_W_flat = fisher_W.flatten()  # [num_classes * C]

    # Compute Fisher-weighted gradient: g^2 / F^p for ALL parameters
    fisher_weighted_grad = (grad_W_flat ** 2) / (fisher_W_flat ** fisher_power + 1e-10)

    # Get top-k parameters (or all if k=-1)
    if k == -1:
        topk_indices = torch.arange(len(fisher_weighted_grad))
        topk_values = fisher_weighted_grad
    else:
        topk_values, topk_indices = torch.topk(fisher_weighted_grad, k=min(k, len(fisher_weighted_grad)), largest=True)

    # Compute gradient w.r.t. feature maps for Grad-CAM
    # Use predicted class logit for gradient computation
    target_logit = logits[0, predicted_class]
    target_logit.backward(retain_graph=True)
    feature_grads = features_grad.grad  # Shape: [1, C, H, W]

    # Compute Fisher-weighted Grad-CAM spatial contributions for each top-k parameter and average
    spatial_contributions = []

    for i, (idx, fisher_grad_val) in enumerate(zip(topk_indices, topk_values)):
        idx = idx.item()

        # Convert flat index to (class, channel)
        param_class = idx // C
        param_channel = idx % C

        # Get spatial feature map for this channel
        channel_feature = features[0, param_channel].detach().cpu().numpy()  # Shape: [H, W]

        # Get Grad-CAM weight: global average pooling of gradients for this channel
        grad_cam_weight = feature_grads[0, param_channel].mean().item()

        # Compute Fisher-weighted Grad-CAM spatial contribution
        # (g^2 / F^p) * (∂y/∂F_i) * F_{i,h,w}
        fisher_weighted_gradcam = fisher_grad_val.item() * grad_cam_weight * channel_feature

        spatial_contributions.append(fisher_weighted_gradcam)

    # Average Fisher-weighted Grad-CAM spatial contributions
    spatial_contribution_avg = np.mean(spatial_contributions, axis=0)  # Shape: [H, W]

    return spatial_contribution_avg


def create_overlay(img_square, spatial_activation, use_colorful=False):
    """
    Create enhanced overlay visualization with per-sample max-min normalization.

    Args:
        img_square: PIL Image (square cropped)
        spatial_activation: numpy array [H, W] with spatial activation values
        use_colorful: bool, if True use turbo colormap (for OOD), else use Reds (for ID)

    Returns:
        overlay: PIL Image with overlay applied
    """
    H, W = spatial_activation.shape
    size = img_square.size[0]  # Assume square

    # Per-sample max-min normalization
    sample_min = spatial_activation.min()
    sample_max = spatial_activation.max()
    sample_range = sample_max - sample_min

    if sample_range > 0:
        activation_gradcam = (spatial_activation - sample_min) / sample_range
    else:
        activation_gradcam = np.zeros_like(spatial_activation)

    # Clip to [0, 1] (negative values will map to lower end of range)
    activation_gradcam = np.clip(activation_gradcam, 0, 1)

    # Resize to match image size with smooth interpolation
    zoom_factor = (size / H, size / W)
    activation_resized = zoom(activation_gradcam, zoom_factor, order=3)  # Cubic

    # Ensure non-negative values before power transform
    activation_resized = np.clip(activation_resized, 0, 1)

    # Apply power transform to emphasize high values
    activation_smooth = activation_resized  # Power transform for emphasis

    # Gentle power transform for better visualization
    activation_enhanced = activation_smooth ** 10

    if use_colorful:
        # Use turbo colormap for OOD: blue-cyan-green-yellow-red
        cmap = matplotlib.colormaps.get_cmap('turbo')
        rgba = cmap(activation_enhanced)

        # Alpha channel: make low values transparent
        alpha_min = 0.7
        alpha_max = 0.8
        alpha_channel = alpha_min + (alpha_max - alpha_min) * (activation_enhanced ** 2)
        rgba[..., 3] = alpha_channel
    else:
        # Use Reds colormap for ID: transparent to red
        cmap = matplotlib.colormaps.get_cmap('turbo')
        rgba = cmap(activation_enhanced)

        # Alpha channel: make low values transparent
        alpha_min = 0.7
        alpha_max = 0.8
        alpha_channel = alpha_min + (alpha_max - alpha_min) * (activation_enhanced ** 2)
        rgba[..., 3] = alpha_channel

    # Convert to PIL Image
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    overlay_img = Image.fromarray(rgba_uint8, mode='RGBA')

    # Composite overlay on original image
    img_rgba = img_square.convert('RGBA')
    result = Image.alpha_composite(img_rgba, overlay_img)

    return result.convert('RGB')


def main():
    # Load model
    print("Loading model...")
    model = get_model()

    # Load Fisher matrix
    fisher_cache_path = os.path.expanduser('~/.cache/openood/fisher_matrices/fisher_imagenet_ResNet50_nll.pt')
    print(f"Loading Fisher matrix from {fisher_cache_path}")
    fisher_W = load_fisher_matrix(fisher_cache_path)

    # Preprocessor
    preprocessor = trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Image paths - only 2 images
    images = [
        ('temp/ID/ILSVRC2012_val_00037071.JPEG', 'ID'),
        ('temp/OOD/quokka-2676171_1280.jpg', 'OOD'),
    ]

    # Process images
    fisher_power = 9.0
    topk = -1

    print("\nComputing spatial activations for all methods...")
    results = []  # List of (img_square, nll_gradcam, kl_gradcam, gtg_gradcam, fisher_gradcam)

    for img_path, label in images:
        print(f"\nProcessing {label}: {os.path.basename(img_path)}")

        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = preprocessor(img_pil).unsqueeze(0)

        feature_extractor = FeatureExtractor(model)
        with torch.no_grad():
            logits, features = feature_extractor(img_tensor)

        pred_class = torch.argmax(logits, dim=1).item()
        fc_weight = model.fc.weight.data
        fc_bias = model.fc.bias.data if model.fc.bias is not None else None

        # Compute four different Grad-CAM methods
        print("  Computing NLL Grad-CAM...")
        nll_gradcam = compute_nll_gradcam(features, fc_weight, fc_bias, pred_class)

        print("  Computing KL-uniform Grad-CAM...")
        kl_gradcam = compute_kl_uniform_gradcam(features, fc_weight, fc_bias, pred_class)

        print("  Computing g^Tg top-k Grad-CAM...")
        gtg_gradcam = compute_gtg_topk_gradcam(features, fc_weight, fc_bias, pred_class, k=topk)

        print("  Computing Fisher-weighted (g^2/F^p) Grad-CAM...")
        fisher_gradcam = compute_fisher_weighted_gradcam(
            features, fc_weight, fc_bias, pred_class, fisher_W, fisher_power, k=topk
        )

        # Center crop image to square
        width, height = img_pil.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        img_square = img_pil.crop((left, top, left + size, top + size))

        # Print statistics
        print(f"  NLL Grad-CAM: min={nll_gradcam.min():.4e}, max={nll_gradcam.max():.4e}, mean={nll_gradcam.mean():.4e}")
        print(f"  KL Grad-CAM: min={kl_gradcam.min():.4e}, max={kl_gradcam.max():.4e}, mean={kl_gradcam.mean():.4e}")
        print(f"  g^Tg Grad-CAM: min={gtg_gradcam.min():.4e}, max={gtg_gradcam.max():.4e}, mean={gtg_gradcam.mean():.4e}")
        print(f"  Fisher Grad-CAM: min={fisher_gradcam.min():.4e}, max={fisher_gradcam.max():.4e}, mean={fisher_gradcam.mean():.4e}")

        results.append((img_square, nll_gradcam, kl_gradcam, gtg_gradcam, fisher_gradcam))

    # Create overlays for all methods
    print("\nCreating overlays with per-sample max-min normalization...")
    all_overlays = []
    for i, (img_square, nll_gradcam, kl_gradcam, gtg_gradcam, fisher_gradcam) in enumerate(results):
        # Use colorful colormap for OOD (index 1), regular for ID (index 0)
        use_colorful = (i == 1)

        nll_overlay = create_overlay(img_square, nll_gradcam, use_colorful=use_colorful)
        kl_overlay = create_overlay(img_square, kl_gradcam, use_colorful=use_colorful)
        gtg_overlay = create_overlay(img_square, gtg_gradcam, use_colorful=use_colorful)
        fisher_overlay = create_overlay(img_square, fisher_gradcam, use_colorful=use_colorful)
        all_overlays.append((img_square, nll_overlay, kl_overlay, gtg_overlay, fisher_overlay))

    # Create Figure 1 layout: 2 rows × 5 columns
    # Each row corresponds to one image (ID or OOD)
    # Columns: [Original, NLL, KL-uniform, g^Tg top-k, g^2/F^p]

    print("\nCreating Figure 1...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Remove all axes
    for ax in axes.flat:
        ax.axis('off')
        ax.set_aspect('equal')

    # Row 0: ID image
    axes[0, 0].imshow(all_overlays[0][0])  # Original
    axes[0, 1].imshow(all_overlays[0][1])  # NLL
    axes[0, 2].imshow(all_overlays[0][2])  # KL-uniform
    axes[0, 3].imshow(all_overlays[0][3])  # g^Tg top-k
    axes[0, 4].imshow(all_overlays[0][4])  # g^2/F^p

    # Row 1: OOD image (quokka)
    axes[1, 0].imshow(all_overlays[1][0])  # Original
    axes[1, 1].imshow(all_overlays[1][1])  # NLL
    axes[1, 2].imshow(all_overlays[1][2])  # KL-uniform
    axes[1, 3].imshow(all_overlays[1][3])  # g^Tg top-k
    axes[1, 4].imshow(all_overlays[1][4])  # g^2/F^p

    # Tight layout with equal spacing (no titles)
    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0, right=1, top=1, bottom=0)

    # Save
    output_path = 'figures/figure1_spatial_activation.pdf'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.05, dpi=300)
    print(f"✓ Saved to {output_path}")

    # Also save PNG
    output_png = output_path.replace('.pdf', '.png')
    plt.savefig(output_png, bbox_inches='tight', pad_inches=0.05, dpi=300)
    print(f"✓ Saved to {output_png}")

    plt.close()


if __name__ == '__main__':
    main()
