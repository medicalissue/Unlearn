"""
Visualize Fisher Information Matrix distribution.

Usage:
    python scripts/visualize_fisher.py \
        --checkpoint results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt \
        --config configs/datasets/cifar10/cifar10.yml \
        --network-config configs/networks/resnet18_32x32.yml \
        --output-dir results/fisher_viz
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openood.networks import get_network
from openood.postprocessors.find_postprocessor import FInDPostprocessor
from openood.utils import Config


def load_fisher_matrix(checkpoint_path, config_path, network_config_path):
    """Load Fisher matrix from postprocessor."""
    # Load configs
    config = Config(config_path)
    net_config = Config(network_config_path)

    # Merge configs
    config.merge_from_other_cfg(net_config)
    config.network.checkpoint = checkpoint_path

    # Create network
    net = get_network(config.network)
    net.load_state_dict(torch.load(checkpoint_path), strict=False)
    net = net.cuda()
    net.eval()

    # Create postprocessor with Fisher computation
    postprocessor_config = Config({
        'postprocessor': {
            'postprocessor_args': {
                'fisher_power': 1.0,
                'fisher_gradient_type': 'nll',
                'test_gradient_type': 'nll',
                'use_adaptive_power': False,
                'use_topk': False,
            }
        },
        'dataset': config.dataset,
        'network': config.network
    })

    postprocessor = FInDPostprocessor(postprocessor_config)

    # Load Fisher matrix (will load from cache if available)
    from openood.datasets import get_dataloader
    id_loader_dict = get_dataloader(config)
    postprocessor.setup(net, id_loader_dict, {})

    # Get Fisher matrix
    fisher_W = postprocessor.fisher_W_tensor.cpu().numpy()  # [num_classes, feature_dim]
    fisher_b = postprocessor.fisher_b_tensor.cpu().numpy() if postprocessor.fisher_b_tensor is not None else None

    return fisher_W, fisher_b, config


def visualize_fisher_distribution(fisher_W, fisher_b, output_dir, config):
    """Create multiple visualizations of Fisher matrix distribution."""
    os.makedirs(output_dir, exist_ok=True)

    # Flatten Fisher matrix
    fisher_flat = fisher_W.flatten()
    if fisher_b is not None:
        fisher_flat = np.concatenate([fisher_flat, fisher_b.flatten()])

    num_classes, feature_dim = fisher_W.shape

    print(f"Fisher matrix shape: {fisher_W.shape}")
    print(f"Total dimensions: {len(fisher_flat)}")
    print(f"Fisher statistics:")
    print(f"  Min: {fisher_flat.min():.6f}")
    print(f"  Max: {fisher_flat.max():.6f}")
    print(f"  Mean: {fisher_flat.mean():.6f}")
    print(f"  Median: {np.median(fisher_flat):.6f}")
    print(f"  Std: {fisher_flat.std():.6f}")

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)

    # 1. Histogram of Fisher values (log scale)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1a. Linear scale histogram
    ax = axes[0, 0]
    ax.hist(fisher_flat, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Fisher Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Fisher Value Distribution (Linear Scale)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 1b. Log scale histogram
    ax = axes[0, 1]
    fisher_nonzero = fisher_flat[fisher_flat > 0]
    ax.hist(np.log10(fisher_nonzero), bins=100, alpha=0.7, color='coral', edgecolor='black')
    ax.set_xlabel('log10(Fisher Value)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Fisher Value Distribution (Log Scale)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 1c. CDF
    ax = axes[1, 0]
    sorted_fisher = np.sort(fisher_flat)
    cdf = np.arange(1, len(sorted_fisher) + 1) / len(sorted_fisher)
    ax.plot(sorted_fisher, cdf, linewidth=2, color='forestgreen')
    ax.set_xlabel('Fisher Value', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 1d. Box plot per class
    ax = axes[1, 1]
    fisher_per_class = [fisher_W[c] for c in range(min(num_classes, 10))]  # Limit to 10 classes
    bp = ax.boxplot(fisher_per_class, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xlabel('Class ID', fontsize=12)
    ax.set_ylabel('Fisher Value', fontsize=12)
    ax.set_title(f'Fisher Distribution per Class (first {min(num_classes, 10)} classes)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fisher_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'fisher_distribution.png')}")
    plt.close()

    # 2. Heatmap of Fisher matrix (per class)
    fig, ax = plt.subplots(figsize=(14, 10))

    # Subsample if too large
    if feature_dim > 100:
        step = feature_dim // 100
        fisher_W_sub = fisher_W[:, ::step]
    else:
        fisher_W_sub = fisher_W

    im = ax.imshow(fisher_W_sub, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Feature Dimension', fontsize=12)
    ax.set_ylabel('Class ID', fontsize=12)
    ax.set_title(f'Fisher Information Matrix Heatmap\n({num_classes} classes × {fisher_W_sub.shape[1]} dims)',
                 fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fisher Value', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fisher_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'fisher_heatmap.png')}")
    plt.close()

    # 3. Top-k analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 3a. Contribution of top-k dimensions
    ax = axes[0]
    sorted_fisher_desc = np.sort(fisher_flat)[::-1]
    cumsum = np.cumsum(sorted_fisher_desc)
    total = cumsum[-1]
    k_values = [10, 50, 100, 500, 1000, 5000, 10000]
    k_values = [k for k in k_values if k < len(fisher_flat)]

    contributions = []
    for k in k_values:
        contrib = cumsum[k-1] / total * 100
        contributions.append(contrib)
        ax.axhline(y=contrib, color='gray', linestyle='--', alpha=0.3)

    ax.plot(k_values, contributions, marker='o', linewidth=2, markersize=8, color='darkblue')
    ax.set_xlabel('Top-k Dimensions', fontsize=12)
    ax.set_ylabel('Cumulative Contribution (%)', fontsize=12)
    ax.set_title('Cumulative Fisher Contribution by Top-k', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    for k, contrib in zip(k_values, contributions):
        ax.annotate(f'{contrib:.1f}%', (k, contrib), textcoords="offset points",
                   xytext=(0, 5), ha='center', fontsize=9)

    # 3b. Top-k vs Bottom-k comparison
    ax = axes[1]
    k_comp = min(1000, len(fisher_flat) // 10)
    top_k = sorted_fisher_desc[:k_comp]
    bottom_k = sorted_fisher_desc[-k_comp:]

    positions = [1, 2]
    bp = ax.boxplot([top_k, bottom_k], positions=positions, widths=0.6, patch_artist=True,
                     labels=[f'Top-{k_comp}', f'Bottom-{k_comp}'])

    colors = ['lightcoral', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Fisher Value', fontsize=12)
    ax.set_title(f'Top-k vs Bottom-k Fisher Values', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics
    ax.text(1, top_k.mean(), f'μ={top_k.mean():.2e}', ha='center', va='bottom', fontsize=10)
    ax.text(2, bottom_k.mean(), f'μ={bottom_k.mean():.2e}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fisher_topk_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'fisher_topk_analysis.png')}")
    plt.close()

    # 4. Per-class statistics
    fig, ax = plt.subplots(figsize=(14, 6))

    class_means = fisher_W.mean(axis=1)
    class_stds = fisher_W.std(axis=1)
    class_mins = fisher_W.min(axis=1)
    class_maxs = fisher_W.max(axis=1)

    x = np.arange(num_classes)
    ax.plot(x, class_means, marker='o', label='Mean', linewidth=2, markersize=4)
    ax.fill_between(x, class_means - class_stds, class_means + class_stds, alpha=0.3, label='±1 Std')
    ax.plot(x, class_mins, linestyle='--', alpha=0.5, label='Min', linewidth=1)
    ax.plot(x, class_maxs, linestyle='--', alpha=0.5, label='Max', linewidth=1)

    ax.set_xlabel('Class ID', fontsize=12)
    ax.set_ylabel('Fisher Value', fontsize=12)
    ax.set_title('Fisher Statistics per Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fisher_per_class.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {os.path.join(output_dir, 'fisher_per_class.png')}")
    plt.close()

    print(f"\n✓ All visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Fisher Information Matrix')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to dataset config')
    parser.add_argument('--network-config', type=str, required=True,
                       help='Path to network config')
    parser.add_argument('--output-dir', type=str, default='results/fisher_viz',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    print("="*80)
    print("Fisher Information Matrix Visualization")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Network Config: {args.network_config}")
    print(f"Output Directory: {args.output_dir}")
    print("="*80)

    # Load Fisher matrix
    print("\nLoading Fisher matrix...")
    fisher_W, fisher_b, config = load_fisher_matrix(
        args.checkpoint, args.config, args.network_config
    )

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_fisher_distribution(fisher_W, fisher_b, args.output_dir, config)

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == '__main__':
    main()
