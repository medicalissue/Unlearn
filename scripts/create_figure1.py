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


def compute_topk_avg_spatial_activation(features, fc_weight, fc_bias, predicted_class, fisher_W, fisher_power=1.0, k=10):
    """
    Compute averaged Fisher-weighted spatial contributions for top-k selected FC weights.

    Returns Fisher-weighted spatial contribution: (|grad_i| / F_i^p) * |F_{i,h,w}|
    This properly shows which spatial regions contribute to high OOD scores.
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

    # Compute Fisher-weighted spatial contributions for each top-k parameter and average
    spatial_contributions = []

    for i, (idx, fisher_grad_val) in enumerate(zip(topk_indices, topk_values)):
        idx = idx.item()

        # Convert flat index to (class, channel)
        param_class = idx // C
        param_channel = idx % C

        # Get spatial feature map for this channel
        channel_feature = features[0, param_channel].cpu().numpy()  # Shape: [H, W]

        # Compute Fisher-weighted spatial contribution
        # fisher_weighted_spatial = (|grad_i| / F_i^p) * |F_{i,h,w}|
        fisher_weighted_spatial = fisher_grad_val.item() * np.abs(channel_feature)

        spatial_contributions.append(fisher_weighted_spatial)

    # Average Fisher-weighted spatial contributions
    spatial_contribution_avg = np.mean(spatial_contributions, axis=0)  # Shape: [H, W]

    return spatial_contribution_avg


def create_overlay(img_square, spatial_activation, all_activations):
    """
    Create enhanced overlay visualization with global absolute scaling.

    Args:
        img_square: PIL Image (square cropped)
        spatial_activation: numpy array [H, W] with spatial activation values
        all_activations: list of all spatial activations for computing global scale

    Returns:
        overlay: PIL Image with overlay applied
    """
    H, W = spatial_activation.shape
    size = img_square.size[0]  # Assume square

    # Use global min/max across all images for consistent scaling
    global_min = min(act.min() for act in all_activations)
    global_max = max(act.max() for act in all_activations)
    global_range = global_max - global_min

    # Normalize using global scale
    activation_gradcam = spatial_activation.copy()
    if global_range > 0:
        activation_gradcam = (activation_gradcam - global_min) / global_range
    else:
        activation_gradcam = np.zeros_like(activation_gradcam)

    # Clip to [0, 1]
    activation_gradcam = np.clip(activation_gradcam, 0, 1)

    # Resize to match image size with smooth interpolation
    zoom_factor = (size / H, size / W)
    activation_resized = zoom(activation_gradcam, zoom_factor, order=3)  # Cubic

    # Apply Gaussian smoothing
    activation_smooth = gaussian_filter(activation_resized, sigma=2.0)

    # Enhance contrast
    activation_enhanced = np.clip(activation_smooth, 0, 1) ** 0.7

    # Create RGBA image with YlOrRd colormap
    cmap = matplotlib.colormaps.get_cmap('YlOrRd')
    rgba = cmap(activation_enhanced)

    # Alpha channel proportional to activation
    alpha_base = 0.7
    alpha_channel = alpha_base * activation_smooth
    rgba[..., 3] = alpha_channel

    # Convert to PIL Image
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    overlay_img = Image.fromarray(rgba_uint8, mode='RGBA')

    # Composite overlay on original image
    img_rgba = img_square.convert('RGBA')
    result = Image.alpha_composite(img_rgba, overlay_img)

    return result.convert('RGB')


def process_image(img_path, model, preprocessor, fisher_W, fisher_power=1.0, topk=10):
    """
    Process a single image and return original + overlay.

    Returns:
        img_square: Square cropped original image (PIL)
        overlay_img: Overlay image (PIL)
    """
    # Load and preprocess image
    img_pil = Image.open(img_path).convert('RGB')
    img_tensor = preprocessor(img_pil).unsqueeze(0)

    # Create feature extractor
    feature_extractor = FeatureExtractor(model)

    # Forward pass
    with torch.no_grad():
        logits, features = feature_extractor(img_tensor)

    # Get prediction
    pred_class = torch.argmax(logits, dim=1).item()

    # Get FC layer weights and bias
    fc_weight = model.fc.weight.data
    fc_bias = model.fc.bias.data if model.fc.bias is not None else None

    # Compute averaged spatial activation for top-k parameters
    spatial_activation = compute_topk_avg_spatial_activation(
        features, fc_weight, fc_bias, pred_class, fisher_W, fisher_power, k=topk
    )

    # Center crop image to square
    width, height = img_pil.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    img_square = img_pil.crop((left, top, left + size, top + size))

    # Create overlay
    overlay_img = create_overlay(img_square, spatial_activation)

    return img_square, overlay_img


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

    # Image paths
    id_images = [
        'temp/ID/ILSVRC2012_val_00036290.JPEG',
        'temp/ID/ILSVRC2012_val_00037071.JPEG',
    ]

    ood_images = [
        'temp/OOD/quokka-2676171_1280.jpg',
        'temp/OOD/shuttlecock_016_0082.png',
    ]

    # Process images
    fisher_power = 9.0
    topk = 10

    # First pass: compute all spatial activations for global scaling
    print("\nComputing spatial activations...")
    all_spatial_activations = []
    all_img_squares = []

    for img_path in id_images + ood_images:
        img_pil = Image.open(img_path).convert('RGB')
        img_tensor = preprocessor(img_pil).unsqueeze(0)

        feature_extractor = FeatureExtractor(model)
        with torch.no_grad():
            logits, features = feature_extractor(img_tensor)

        pred_class = torch.argmax(logits, dim=1).item()
        fc_weight = model.fc.weight.data
        fc_bias = model.fc.bias.data if model.fc.bias is not None else None

        spatial_activation = compute_topk_avg_spatial_activation(
            features, fc_weight, fc_bias, pred_class, fisher_W, fisher_power, k=topk
        )

        # Center crop image to square
        width, height = img_pil.size
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        img_square = img_pil.crop((left, top, left + size, top + size))

        all_spatial_activations.append(spatial_activation)
        all_img_squares.append(img_square)

    # Second pass: create overlays with global scaling
    print("\nCreating overlays with global scaling...")
    id_results = []
    for i, img_square in enumerate(all_img_squares[:len(id_images)]):
        overlay = create_overlay(img_square, all_spatial_activations[i], all_spatial_activations)
        id_results.append((img_square, overlay))

    ood_results = []
    for i, img_square in enumerate(all_img_squares[len(id_images):]):
        overlay = create_overlay(img_square, all_spatial_activations[len(id_images) + i], all_spatial_activations)
        ood_results.append((img_square, overlay))

    # Create Figure 1 layout: 2 rows × 4 columns
    # Row 1: ID1 orig, ID1 overlay, OOD1 orig, OOD1 overlay
    # Row 2: ID2 orig, ID2 overlay, OOD2 orig, OOD2 overlay

    print("\nCreating Figure 1...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Remove all axes
    for ax in axes.flat:
        ax.axis('off')
        ax.set_aspect('equal')

    # Row 0: First ID and OOD pair
    axes[0, 0].imshow(id_results[0][0])  # ID1 original
    axes[0, 1].imshow(id_results[0][1])  # ID1 overlay
    axes[0, 2].imshow(ood_results[0][0])  # OOD1 original
    axes[0, 3].imshow(ood_results[0][1])  # OOD1 overlay

    # Row 1: Second ID and OOD pair
    axes[1, 0].imshow(id_results[1][0])  # ID2 original
    axes[1, 1].imshow(id_results[1][1])  # ID2 overlay
    axes[1, 2].imshow(ood_results[1][0])  # OOD2 original
    axes[1, 3].imshow(ood_results[1][1])  # OOD2 overlay

    # Tight layout with minimal spacing
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
