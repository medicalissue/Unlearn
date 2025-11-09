"""
Simple Fisher Matrix Visualization from Cache

Usage:
    python scripts/viz_fisher_simple.py \
        --fisher-cache path/to/fisher_matrices/fisher_cifar10_ResNet18_32x32.pt \
        --output-dir fisher_viz
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from scipy import stats


def load_fisher_from_cache(cache_path):
    """Load Fisher matrix from cache file."""
    print(f"Loading Fisher cache: {cache_path}")
    cache = torch.load(cache_path, map_location='cpu')

    fisher_W = cache['fisher_W'].numpy()  # [num_classes, feature_dim]
    fisher_b = cache['fisher_b'].numpy() if cache['fisher_b'] is not None else None

    print(f"Dataset: {cache.get('dataset_name', 'unknown')}")
    print(f"Model: {cache.get('model_arch', 'unknown')}")
    print(f"Num classes: {cache.get('num_classes', 'unknown')}")
    print(f"Feature dim: {cache.get('feature_dim', 'unknown')}")
    print(f"Fisher W shape: {fisher_W.shape}")
    if fisher_b is not None:
        print(f"Fisher b shape: {fisher_b.shape}")

    return fisher_W, fisher_b, cache


def print_detailed_statistics(fisher_W, fisher_b):
    """Print comprehensive Fisher statistics."""
    # Flatten
    fisher_flat = fisher_W.flatten()
    if fisher_b is not None:
        fisher_flat = np.concatenate([fisher_flat, fisher_b.flatten()])

    num_classes, feature_dim = fisher_W.shape

    print("\n" + "="*80)
    print("FISHER MATRIX DETAILED STATISTICS")
    print("="*80)

    # Basic info
    print(f"\n[Matrix Shape]")
    print(f"  Weight matrix: {fisher_W.shape} (num_classes × feature_dim)")
    if fisher_b is not None:
        print(f"  Bias vector: {fisher_b.shape}")
    print(f"  Total dimensions: {len(fisher_flat):,}")

    # Central tendency
    print(f"\n[Central Tendency]")
    print(f"  Mean:       {fisher_flat.mean():.6e}")
    print(f"  Median:     {np.median(fisher_flat):.6e}")
    print(f"  Mode (approx): {stats.mode(np.round(fisher_flat, 6), keepdims=True)[0][0]:.6e}")
    print(f"  Geometric mean: {stats.gmean(fisher_flat[fisher_flat > 0]):.6e}")

    # Spread
    print(f"\n[Spread]")
    print(f"  Min:        {fisher_flat.min():.6e}")
    print(f"  Max:        {fisher_flat.max():.6e}")
    print(f"  Range:      {fisher_flat.max() - fisher_flat.min():.6e}")
    print(f"  Std:        {fisher_flat.std():.6e}")
    print(f"  Variance:   {fisher_flat.var():.6e}")
    print(f"  IQR:        {np.percentile(fisher_flat, 75) - np.percentile(fisher_flat, 25):.6e}")

    # Shape
    print(f"\n[Distribution Shape]")
    print(f"  Skewness:   {stats.skew(fisher_flat):.6f}")
    skew_interp = "right-skewed (long tail →)" if stats.skew(fisher_flat) > 0 else "left-skewed (long tail ←)" if stats.skew(fisher_flat) < 0 else "symmetric"
    print(f"              ({skew_interp})")
    print(f"  Kurtosis:   {stats.kurtosis(fisher_flat):.6f}")
    kurt_interp = "heavy-tailed (outlier-prone)" if stats.kurtosis(fisher_flat) > 0 else "light-tailed" if stats.kurtosis(fisher_flat) < 0 else "normal-like"
    print(f"              ({kurt_interp})")

    # Percentiles
    print(f"\n[Percentiles]")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(fisher_flat, p)
        print(f"  {p:2d}th:      {val:.6e}")

    # Zero/near-zero analysis
    print(f"\n[Zero/Near-Zero Analysis]")
    n_zero = np.sum(fisher_flat == 0)
    n_near_zero = np.sum(np.abs(fisher_flat) < 1e-10)
    n_small = np.sum(np.abs(fisher_flat) < 1e-6)
    print(f"  Exactly zero:        {n_zero:,} ({n_zero/len(fisher_flat)*100:.2f}%)")
    print(f"  Near-zero (<1e-10):  {n_near_zero:,} ({n_near_zero/len(fisher_flat)*100:.2f}%)")
    print(f"  Small (<1e-6):       {n_small:,} ({n_small/len(fisher_flat)*100:.2f}%)")

    # Top-k analysis
    print(f"\n[Top-k Contribution Analysis]")
    sorted_desc = np.sort(fisher_flat)[::-1]
    cumsum = np.cumsum(sorted_desc)
    total = cumsum[-1]

    k_values = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
    k_values = [k for k in k_values if k < len(fisher_flat)]

    for k in k_values:
        contrib = cumsum[k-1] / total * 100
        print(f"  Top-{k:5d}: {contrib:6.2f}% (value threshold: {sorted_desc[k-1]:.6e})")

    # Class-wise statistics
    print(f"\n[Per-Class Statistics]")
    class_means = fisher_W.mean(axis=1)
    class_stds = fisher_W.std(axis=1)
    class_mins = fisher_W.min(axis=1)
    class_maxs = fisher_W.max(axis=1)

    print(f"  Class with highest mean: {class_means.argmax()} (mean={class_means.max():.6e})")
    print(f"  Class with lowest mean:  {class_means.argmin()} (mean={class_means.min():.6e})")
    print(f"  Class with highest std:  {class_stds.argmax()} (std={class_stds.max():.6e})")
    print(f"  Class with lowest std:   {class_stds.argmin()} (std={class_stds.min():.6e})")

    # Dimension-wise statistics
    print(f"\n[Per-Dimension Statistics]")
    dim_means = fisher_W.mean(axis=0)
    dim_stds = fisher_W.std(axis=0)

    print(f"  Dimension with highest mean: {dim_means.argmax()} (mean={dim_means.max():.6e})")
    print(f"  Dimension with lowest mean:  {dim_means.argmin()} (mean={dim_means.min():.6e})")
    print(f"  Dimension with highest std:  {dim_stds.argmax()} (std={dim_stds.max():.6e})")
    print(f"  Dimension with lowest std:   {dim_stds.argmin()} (std={dim_stds.min():.6e})")

    # Concentration analysis
    print(f"\n[Concentration Analysis]")
    gini = calculate_gini(fisher_flat)
    print(f"  Gini coefficient: {gini:.6f}")
    gini_interp = "highly concentrated" if gini > 0.6 else "moderately concentrated" if gini > 0.3 else "fairly uniform"
    print(f"                    ({gini_interp})")

    # Effective dimensionality
    eigensum = fisher_flat.sum()
    eigensum_sq = (fisher_flat ** 2).sum()
    eff_dim = eigensum ** 2 / eigensum_sq if eigensum_sq > 0 else 0
    print(f"  Effective dimensionality: {eff_dim:.2f} / {len(fisher_flat):,}")
    print(f"                            ({eff_dim/len(fisher_flat)*100:.2f}% of total)")

    print("="*80 + "\n")

    return fisher_flat


def calculate_gini(array):
    """Calculate Gini coefficient for concentration measure."""
    array = np.abs(array)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)) if np.sum(array) > 0 else 0


def visualize_fisher(fisher_W, fisher_b, output_dir, cache_name):
    """Create Fisher distribution visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # Print detailed statistics
    fisher_flat = print_detailed_statistics(fisher_W, fisher_b)

    num_classes, feature_dim = fisher_W.shape

    sns.set_style("whitegrid")

    # Figure 1: Distribution overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Histogram (linear)
    ax = axes[0, 0]
    ax.hist(fisher_flat, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Fisher Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution (Linear Scale)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Histogram (log)
    ax = axes[0, 1]
    fisher_pos = fisher_flat[fisher_flat > 0]
    ax.hist(np.log10(fisher_pos), bins=100, alpha=0.7, color='coral', edgecolor='black')
    ax.set_xlabel('log10(Fisher Value)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution (Log Scale)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # CDF
    ax = axes[1, 0]
    sorted_fisher = np.sort(fisher_flat)
    cdf = np.arange(1, len(sorted_fisher) + 1) / len(sorted_fisher)
    ax.plot(sorted_fisher, cdf, linewidth=2, color='forestgreen')
    ax.set_xlabel('Fisher Value', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Per-class boxplot
    ax = axes[1, 1]
    n_show = min(10, num_classes)
    fisher_per_class = [fisher_W[c] for c in range(n_show)]
    bp = ax.boxplot(fisher_per_class, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_xlabel('Class ID', fontsize=12)
    ax.set_ylabel('Fisher Value', fontsize=12)
    ax.set_title(f'Per-Class Distribution (first {n_show})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{cache_name}_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {save_path}")
    plt.close()

    # Figure 2: Top-k Analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Top-k cumulative contribution
    ax = axes[0]
    sorted_desc = np.sort(fisher_flat)[::-1]
    cumsum = np.cumsum(sorted_desc)
    total = cumsum[-1]

    k_values = [10, 50, 100, 500, 1000, 5000, 10000]
    k_values = [k for k in k_values if k < len(fisher_flat)]

    contributions = []
    for k in k_values:
        contrib = cumsum[k-1] / total * 100
        contributions.append(contrib)

    ax.plot(k_values, contributions, marker='o', linewidth=2, markersize=8, color='darkblue')
    ax.set_xlabel('Top-k Dimensions', fontsize=12)
    ax.set_ylabel('Cumulative Contribution (%)', fontsize=12)
    ax.set_title('Top-k Fisher Contribution', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    for k, contrib in zip(k_values, contributions):
        ax.annotate(f'{contrib:.1f}%', (k, contrib), textcoords="offset points",
                   xytext=(0, 5), ha='center', fontsize=9)

    # Top vs Bottom
    ax = axes[1]
    k_comp = min(1000, len(fisher_flat) // 10)
    top_k = sorted_desc[:k_comp]
    bottom_k = sorted_desc[-k_comp:]

    bp = ax.boxplot([top_k, bottom_k], positions=[1, 2], widths=0.6, patch_artist=True,
                     labels=[f'Top-{k_comp}', f'Bottom-{k_comp}'])

    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')

    ax.set_ylabel('Fisher Value', fontsize=12)
    ax.set_title('Top-k vs Bottom-k', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    ax.text(1, top_k.mean(), f'μ={top_k.mean():.2e}', ha='center', va='bottom', fontsize=10)
    ax.text(2, bottom_k.mean(), f'μ={bottom_k.mean():.2e}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{cache_name}_topk.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

    # Figure 3: Heatmap
    fig, ax = plt.subplots(figsize=(14, 10))

    # Subsample if too large
    if feature_dim > 100:
        step = max(1, feature_dim // 100)
        fisher_sub = fisher_W[:, ::step]
    else:
        fisher_sub = fisher_W

    im = ax.imshow(fisher_sub, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Feature Dimension', fontsize=12)
    ax.set_ylabel('Class ID', fontsize=12)
    ax.set_title(f'Fisher Matrix Heatmap ({num_classes} × {fisher_sub.shape[1]})',
                 fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fisher Value', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{cache_name}_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

    # Figure 4: Power-law analysis (Log-Log plot)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Log-Log plot: Rank vs Value
    ax = axes[0]
    sorted_desc = np.sort(fisher_flat)[::-1]
    rank = np.arange(1, len(sorted_desc) + 1)

    # Filter positive values for log scale
    mask = sorted_desc > 0
    rank_filtered = rank[mask]
    values_filtered = sorted_desc[mask]

    ax.loglog(rank_filtered, values_filtered, 'o', alpha=0.5, markersize=3, color='darkblue')
    ax.set_xlabel('Rank (log scale)', fontsize=12)
    ax.set_ylabel('Fisher Value (log scale)', fontsize=12)
    ax.set_title('Power-Law Analysis: Rank vs Value (Log-Log)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Fit power law to the top portion (first 10%)
    n_fit = max(100, len(rank_filtered) // 10)
    if n_fit > 10:
        log_rank = np.log10(rank_filtered[:n_fit])
        log_values = np.log10(values_filtered[:n_fit])

        # Linear fit in log-log space: log(y) = a + b*log(x)
        coeffs = np.polyfit(log_rank, log_values, 1)
        slope, intercept = coeffs[0], coeffs[1]

        # Plot fitted line
        fit_x = rank_filtered[:n_fit]
        fit_y = 10**(intercept + slope * np.log10(fit_x))
        ax.loglog(fit_x, fit_y, 'r--', linewidth=2, label=f'Power law fit: α={-slope:.3f}')
        ax.legend(fontsize=10)

        # Print power-law exponent
        print(f"\n[Power-Law Analysis]")
        print(f"  Fitted exponent α: {-slope:.6f}")
        print(f"  (fitted on top {n_fit:,} values)")
        if abs(slope + 1) < 0.2:
            print(f"  Interpretation: Close to Zipf's law (α ≈ 1)")
        elif -slope < 1:
            print(f"  Interpretation: Gentle power law (heavy-headed)")
        elif -slope > 2:
            print(f"  Interpretation: Steep power law (many small values)")
        else:
            print(f"  Interpretation: Moderate power law")

    # Log-Log plot: CCDF (Complementary Cumulative Distribution)
    ax = axes[1]
    sorted_vals = np.sort(fisher_flat[fisher_flat > 0])
    ccdf = 1.0 - np.arange(len(sorted_vals)) / len(sorted_vals)

    ax.loglog(sorted_vals, ccdf, 'o', alpha=0.5, markersize=3, color='forestgreen')
    ax.set_xlabel('Fisher Value (log scale)', fontsize=12)
    ax.set_ylabel('P(X > x) - CCDF (log scale)', fontsize=12)
    ax.set_title('Power-Law: CCDF (Complementary CDF)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Fit power law to CCDF tail
    n_tail = max(100, len(sorted_vals) // 10)
    if n_tail > 10:
        tail_vals = sorted_vals[-n_tail:]
        tail_ccdf = ccdf[-n_tail:]

        log_vals = np.log10(tail_vals)
        log_ccdf = np.log10(tail_ccdf)

        coeffs_ccdf = np.polyfit(log_vals, log_ccdf, 1)
        slope_ccdf, intercept_ccdf = coeffs_ccdf[0], coeffs_ccdf[1]

        fit_x_ccdf = tail_vals
        fit_y_ccdf = 10**(intercept_ccdf + slope_ccdf * np.log10(fit_x_ccdf))
        ax.loglog(fit_x_ccdf, fit_y_ccdf, 'r--', linewidth=2,
                 label=f'Power law fit: exponent={-slope_ccdf:.3f}')
        ax.legend(fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{cache_name}_powerlaw.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

    print(f"\n✓ All visualizations saved to: {output_dir}")
    print(f"  Prefix: {cache_name}")
    print(f"  Files: {cache_name}_distribution.png, {cache_name}_topk.png, {cache_name}_heatmap.png, {cache_name}_powerlaw.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize Fisher Matrix from Cache')
    parser.add_argument('--fisher-cache', type=str, required=True,
                       help='Path to Fisher cache file (.pt)')
    parser.add_argument('--output-dir', type=str, default='fisher_viz',
                       help='Output directory')

    args = parser.parse_args()

    print("="*80)
    print("Fisher Matrix Visualization")
    print("="*80)

    # Extract cache name (without .pt extension)
    cache_basename = os.path.basename(args.fisher_cache)
    cache_name = os.path.splitext(cache_basename)[0]  # Remove .pt

    # Load
    fisher_W, fisher_b, cache = load_fisher_from_cache(args.fisher_cache)

    # Visualize
    visualize_fisher(fisher_W, fisher_b, args.output_dir, cache_name)

    print("\n" + "="*80)
    print("Done!")
    print("="*80)


if __name__ == '__main__':
    main()
