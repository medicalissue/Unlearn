#!/usr/bin/env python
"""
Analyze and visualize Top-k parameter selection patterns.

Usage:
    python scripts/analyze_topk_patterns.py \
        --indices topk_indices.txt \
        --num-classes 1000 \
        --feature-dim 2048 \
        --output figures/
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path


def decode_indices(indices, num_classes, feature_dim):
    """Decode flattened indices to (class, feature) tuples."""
    weight_param_count = num_classes * feature_dim

    weight_indices = []
    bias_indices = []
    weight_decoded = []

    for idx in indices:
        if idx < weight_param_count:
            # Weight parameter
            weight_indices.append(idx)
            class_idx = idx // feature_dim
            feat_idx = idx % feature_dim
            weight_decoded.append((class_idx, feat_idx))
        else:
            # Bias parameter
            bias_idx = idx - weight_param_count
            bias_indices.append(bias_idx)

    return weight_indices, bias_indices, weight_decoded


def plot_class_distribution(weight_decoded, num_classes, output_dir):
    """Plot histogram of selected parameters per class."""
    class_counts = Counter([c for c, f in weight_decoded])

    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 5))

    # All classes
    classes = list(range(num_classes))
    counts = [class_counts.get(c, 0) for c in classes]

    ax.bar(classes, counts, width=1.0, edgecolor='none', alpha=0.7)
    ax.set_xlabel('Class Index', fontsize=12)
    ax.set_ylabel('Number of Selected Parameters', fontsize=12)
    ax.set_title('Distribution of Selected Parameters Across Classes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Statistics text
    non_zero = sum(1 for c in counts if c > 0)
    max_count = max(counts) if counts else 0
    mean_count = np.mean([c for c in counts if c > 0]) if non_zero > 0 else 0

    stats_text = f'Classes touched: {non_zero}/{num_classes}\n'
    stats_text += f'Max selections: {max_count}\n'
    stats_text += f'Mean (non-zero): {mean_count:.1f}'

    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'class_distribution.pdf'}")
    plt.close()


def plot_feature_distribution(weight_decoded, feature_dim, output_dir):
    """Plot histogram of selected parameters per feature dimension."""
    feature_counts = Counter([f for c, f in weight_decoded])

    # Create histogram
    fig, ax = plt.subplots(figsize=(12, 5))

    # All features
    features = list(range(feature_dim))
    counts = [feature_counts.get(f, 0) for f in features]

    ax.bar(features, counts, width=1.0, edgecolor='none', alpha=0.7, color='coral')
    ax.set_xlabel('Feature Dimension', fontsize=12)
    ax.set_ylabel('Number of Selected Parameters', fontsize=12)
    ax.set_title('Distribution of Selected Parameters Across Feature Dimensions', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Statistics text
    non_zero = sum(1 for c in counts if c > 0)
    max_count = max(counts) if counts else 0
    mean_count = np.mean([c for c in counts if c > 0]) if non_zero > 0 else 0

    stats_text = f'Features touched: {non_zero}/{feature_dim}\n'
    stats_text += f'Max selections: {max_count}\n'
    stats_text += f'Mean (non-zero): {mean_count:.1f}'

    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'feature_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'feature_distribution.pdf'}")
    plt.close()


def plot_2d_heatmap(weight_decoded, num_classes, feature_dim, output_dir):
    """Plot 2D heatmap of (class, feature) selection frequency."""
    # Create matrix
    selection_matrix = np.zeros((num_classes, feature_dim))

    for class_idx, feat_idx in weight_decoded:
        selection_matrix[class_idx, feat_idx] += 1

    # Subsample for visualization if too large
    if num_classes > 200 or feature_dim > 500:
        # Show top classes and features
        class_sums = selection_matrix.sum(axis=1)
        feature_sums = selection_matrix.sum(axis=0)

        top_classes = np.argsort(class_sums)[-100:]
        top_features = np.argsort(feature_sums)[-200:]

        selection_matrix = selection_matrix[top_classes][:, top_features]
        title_suffix = ' (Top 100 Classes × Top 200 Features)'
    else:
        title_suffix = ''

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(selection_matrix, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Feature Dimension', fontsize=12)
    ax.set_ylabel('Class Index', fontsize=12)
    ax.set_title(f'Selection Frequency Heatmap{title_suffix}', fontsize=14, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Selection Count')

    plt.tight_layout()
    plt.savefig(output_dir / 'selection_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'selection_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'selection_heatmap.pdf'}")
    plt.close()


def plot_clustering_analysis(weight_indices, output_dir):
    """Analyze and visualize index clustering."""
    sorted_indices = sorted(weight_indices)

    if len(sorted_indices) < 2:
        print("Not enough indices for clustering analysis")
        return

    # Compute gaps
    gaps = [sorted_indices[i+1] - sorted_indices[i] for i in range(len(sorted_indices)-1)]

    # Plot gap histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gap distribution (log scale)
    ax = axes[0]
    ax.hist(gaps, bins=50, edgecolor='black', alpha=0.7, log=True)
    ax.set_xlabel('Gap Size (log scale)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Gaps Between Selected Indices', fontsize=13, fontweight='bold')
    ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Adjacent (gap=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Cumulative gap distribution
    ax = axes[1]
    sorted_gaps = sorted(gaps)
    cumulative = np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps) * 100
    ax.plot(sorted_gaps, cumulative, linewidth=2)
    ax.set_xlabel('Gap Size', fontsize=12)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax.set_title('Cumulative Gap Distribution', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Add statistics
    adjacent_count = sum(1 for g in gaps if g == 1)
    stats_text = f'Adjacent pairs: {adjacent_count} ({100*adjacent_count/len(gaps):.1f}%)\n'
    stats_text += f'Median gap: {np.median(gaps):.0f}\n'
    stats_text += f'Mean gap: {np.mean(gaps):.0f}'

    ax.text(0.98, 0.05, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'clustering_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'clustering_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'clustering_analysis.pdf'}")
    plt.close()


def load_indices_from_csv(csv_file):
    """Load indices from CSV file (k*_indices.csv format)."""
    indices = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            indices.append(int(row['index']))

    return indices


def main():
    parser = argparse.ArgumentParser(description='Analyze Top-k parameter selection patterns from CSV')
    parser.add_argument('--input', type=str, required=True,
                       help='CSV file containing indices (k*_indices.csv) or directory with CSV files')
    parser.add_argument('--num-classes', type=int, default=1000,
                       help='Number of output classes')
    parser.add_argument('--feature-dim', type=int, default=2048,
                       help='Feature dimension')
    parser.add_argument('--output', type=str, default='figures/',
                       help='Output directory for figures')
    args = parser.parse_args()

    # Check if input is a directory or file
    input_path = Path(args.input)

    if input_path.is_dir():
        # Find the most recent k*_indices.csv file
        csv_files = list(input_path.glob('k*_indices.csv'))
        if not csv_files:
            raise FileNotFoundError(f"No k*_indices.csv files found in {args.input}")

        # Sort by modification time, get most recent
        csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"Found {len(csv_files)} CSV files, using most recent: {csv_file.name}")
    else:
        csv_file = input_path

    # Load indices from CSV
    print(f"Loading indices from: {csv_file}")
    indices = load_indices_from_csv(csv_file)
    print(f"Loaded {len(indices)} unique indices")

    # Decode
    weight_indices, bias_indices, weight_decoded = decode_indices(
        indices, args.num_classes, args.feature_dim
    )

    print(f"Weight parameters: {len(weight_indices)}")
    print(f"Bias parameters: {len(bias_indices)}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    if weight_decoded:
        print("\nGenerating visualizations...")
        plot_class_distribution(weight_decoded, args.num_classes, output_dir)
        plot_feature_distribution(weight_decoded, args.feature_dim, output_dir)
        plot_2d_heatmap(weight_decoded, args.num_classes, args.feature_dim, output_dir)
        plot_clustering_analysis(weight_indices, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
