#!/usr/bin/env python
"""
Plot AUROC vs. Top-k curves for OOD detection parameter sparsity analysis.

Reads pre-computed CSV files from topk_analysis/ directory.

Usage:
    python scripts/plot_topk_results.py --input topk_analysis/ --output figures/
"""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import re


def find_summary_files(input_dir):
    """Find all k*_summary.csv files in the input directory."""
    input_path = Path(input_dir)
    summary_files = list(input_path.glob('k*_summary.csv'))

    if not summary_files:
        raise FileNotFoundError(f"No summary CSV files found in {input_dir}")

    print(f"Found {len(summary_files)} summary files")
    return summary_files


def load_results_from_summaries(summary_files):
    """Load results from individual k*_summary.csv files."""
    results = {}

    for summary_file in summary_files:
        # Extract k value from filename
        match = re.search(r'k(\d+)_', summary_file.name)
        if not match:
            print(f"Warning: Could not extract k value from {summary_file.name}")
            continue

        k = int(match.group(1))

        # Read summary file
        summary_data = {}
        with open(summary_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                summary_data[row['metric']] = row['value']

        # Store k and its data
        results[k] = summary_data
        print(f"  Loaded k={k}: {summary_file.name}")

    return results


def load_auroc_from_logs(log_dir):
    """
    Load AUROC values from evaluation log files.

    Expected format: topk_k{N}.log files containing AUROC metrics.
    """
    log_path = Path(log_dir)
    log_files = list(log_path.glob('topk_k*.log'))

    if not log_files:
        print(f"Warning: No log files found in {log_dir}")
        return {}

    auroc_results = {}

    for log_file in log_files:
        # Extract k value
        if 'topk_all' in log_file.name:
            k = 'ALL'
        else:
            match = re.search(r'topk_k(\d+)', log_file.name)
            if not match:
                continue
            k = int(match.group(1))

        # Parse AUROC values
        with open(log_file, 'r') as f:
            content = f.read()

        auroc_dict = {}
        for line in content.split('\n'):
            if 'AUROC' in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'AUROC' in part and i > 0:
                        dataset = parts[i-1].strip(':')
                        try:
                            auroc = float(parts[i+1].strip('%,'))
                            auroc_dict[dataset] = auroc
                        except:
                            pass

        if auroc_dict:
            auroc_results[k] = auroc_dict

    return auroc_results


def plot_parameter_statistics(results, output_dir, style='paper'):
    """
    Plot parameter selection statistics from summary CSV files.

    Args:
        results: Dict mapping k -> {metric: value}
        output_dir: Path to save figures
        style: 'paper' or 'presentation'
    """
    k_values = sorted(results.keys())

    # Extract metrics
    unique_params = [float(results[k].get('unique_params_selected', 0)) for k in k_values]
    coverage = [float(results[k].get('coverage_percent', 0)) for k in k_values]
    unique_classes = [float(results[k].get('unique_classes_touched', 0)) for k in k_values]
    unique_features = [float(results[k].get('unique_features_touched', 0)) for k in k_values]

    # Style settings
    if style == 'paper':
        plt.rcParams.update({
            'font.size': 10,
            'figure.figsize': (8, 5),
            'lines.linewidth': 2,
            'lines.markersize': 6,
        })
    else:
        plt.rcParams.update({
            'font.size': 14,
            'figure.figsize': (10, 6),
            'lines.linewidth': 3,
            'lines.markersize': 8,
        })

    # Plot: Unique parameters vs k
    fig, ax = plt.subplots()
    ax.plot(k_values, unique_params, marker='o', linewidth=2.5, color='steelblue')
    ax.plot([k_values[0], k_values[-1]], [k_values[0], k_values[-1]], '--',
            color='gray', alpha=0.5, label='y=k (ideal)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('k (parameters per sample)', fontsize=12)
    ax.set_ylabel('Unique parameters selected', fontsize=12)
    ax.set_title('Parameter Sparsity: Coverage vs. k', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'param_coverage_vs_k.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'param_coverage_vs_k.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'param_coverage_vs_k.pdf'}")
    plt.close()

    # Plot: Classes and Features touched
    fig, ax = plt.subplots()
    ax.plot(k_values, unique_classes, marker='o', label='Classes touched', color='coral')
    ax.plot(k_values, unique_features, marker='s', label='Features touched', color='teal')
    ax.axhline(y=1000, color='coral', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=2048, color='teal', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('k (parameters per sample)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Diversity: Classes and Features Touched', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'diversity_vs_k.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'diversity_vs_k.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'diversity_vs_k.pdf'}")
    plt.close()


def plot_topk_curves(results, datasets, output_dir, style='paper'):
    """
    Plot AUROC vs. Top-k curves.

    Args:
        results: Dict mapping k -> {dataset: auroc}
        datasets: List of dataset names
        output_dir: Path to save figures
        style: 'paper' or 'presentation'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort k values
    k_values = sorted([k for k in results.keys() if k != float('inf')])
    if float('inf') in results:
        k_values.append(float('inf'))

    # Style settings
    if style == 'paper':
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.figsize': (6, 4),
            'lines.linewidth': 2,
            'lines.markersize': 6,
        })
    else:
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            'legend.fontsize': 12,
            'figure.figsize': (10, 6),
            'lines.linewidth': 3,
            'lines.markersize': 8,
        })

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))

    # Main figure: AUROC vs. k
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, dataset in enumerate(datasets):
        auroc_values = []
        x_values = []

        for k in k_values:
            if dataset in results[k]:
                auroc_values.append(results[k][dataset])
                x_values.append(k if k != float('inf') else k_values[-2] * 2)

        if auroc_values:
            # Plot line
            ax.plot(x_values[:-1], auroc_values[:-1],
                   marker='o', label=dataset, color=colors[i], linewidth=2.5)

            # Add "ALL" point separately (dashed line)
            if len(x_values) > 1:
                ax.plot([x_values[-2], x_values[-1]], [auroc_values[-2], auroc_values[-1]],
                       '--', color=colors[i], alpha=0.5, linewidth=1.5)
                ax.plot(x_values[-1], auroc_values[-1], 'D', color=colors[i],
                       markersize=8, markeredgecolor='black', markeredgewidth=1)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Top-k Parameters (log scale)', fontsize=13)
    ax.set_ylabel('AUROC (%)', fontsize=13)
    ax.set_title('OOD Detection Performance vs. Parameter Sparsity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.9)

    # Add annotation for k=1
    if 1 in k_values:
        mean_auroc_k1 = np.mean([results[1][d] for d in datasets if d in results[1]])
        mean_auroc_all = np.mean([results[float('inf')][d] for d in datasets if d in results[float('inf')]])
        retention = 100 * mean_auroc_k1 / mean_auroc_all
        ax.annotate(f'k=1: {retention:.1f}% of ALL performance',
                   xy=(1, mean_auroc_k1), xytext=(5, mean_auroc_k1 - 5),
                   fontsize=10, color='red', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'topk_auroc_curve.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'topk_auroc_curve.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'topk_auroc_curve.pdf'}")
    plt.close()

    # Secondary figure: Normalized performance
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, dataset in enumerate(datasets):
        auroc_values = []
        x_values = []

        for k in k_values:
            if dataset in results[k]:
                auroc_values.append(results[k][dataset])
                x_values.append(k if k != float('inf') else k_values[-2] * 2)

        if auroc_values and len(auroc_values) > 1:
            # Normalize by ALL performance
            auroc_all = auroc_values[-1]
            normalized = [100 * a / auroc_all for a in auroc_values]

            ax.plot(x_values[:-1], normalized[:-1],
                   marker='o', label=dataset, color=colors[i], linewidth=2.5)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Top-k Parameters (log scale)', fontsize=13)
    ax.set_ylabel('Normalized Performance (% of ALL)', fontsize=13)
    ax.set_title('Performance Retention with Sparse Parameters', fontsize=14, fontweight='bold')
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.5, label='ALL parameters')
    ax.axhline(y=95, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='95% retention')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_dir / 'topk_normalized.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'topk_normalized.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'topk_normalized.pdf'}")
    plt.close()

    # Print statistics
    print("\n" + "="*80)
    print("Top-k Performance Statistics")
    print("="*80)

    for k in [1, 10, 100, 1000]:
        if k in results:
            print(f"\nTop-{k}:")
            for dataset in datasets:
                if dataset in results[k] and dataset in results[float('inf')]:
                    auroc_k = results[k][dataset]
                    auroc_all = results[float('inf')][dataset]
                    retention = 100 * auroc_k / auroc_all
                    print(f"  {dataset:20s}: {auroc_k:6.2f}% (retention: {retention:5.2f}%)")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Plot Top-k parameter sparsity curves from CSV')
    parser.add_argument('--input', type=str, default='topk_analysis/',
                       help='Input directory containing k*_summary.csv files')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='Optional: Directory containing topk_k*.log files for AUROC data')
    parser.add_argument('--output', type=str, default='figures/',
                       help='Output directory for figures')
    parser.add_argument('--style', type=str, default='paper', choices=['paper', 'presentation'],
                       help='Figure style')
    args = parser.parse_args()

    # Load parameter selection statistics from CSV
    print(f"Loading parameter statistics from: {args.input}")
    summary_files = find_summary_files(args.input)
    results = load_results_from_summaries(summary_files)
    print(f"Loaded {len(results)} k-values")

    # Optional: Load AUROC data from logs
    auroc_results = {}
    if args.log_dir:
        print(f"\nLoading AUROC data from: {args.log_dir}")
        auroc_results = load_auroc_from_logs(args.log_dir)
        print(f"Loaded AUROC for {len(auroc_results)} k-values")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save parameter statistics to CSV before plotting
    if results:
        csv_file = output_dir / 'parameter_statistics.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = ['k', 'unique_params_selected', 'coverage_percent',
                     'unique_classes_touched', 'unique_features_touched',
                     'total_params', 'num_samples']
            writer.writerow(header)

            # Data rows
            for k in sorted(results.keys()):
                row = [
                    k,
                    results[k].get('unique_params_selected', ''),
                    results[k].get('coverage_percent', ''),
                    results[k].get('unique_classes_touched', ''),
                    results[k].get('unique_features_touched', ''),
                    results[k].get('total_params', ''),
                    results[k].get('num_samples', '')
                ]
                writer.writerow(row)

        print(f"Saved parameter statistics to: {csv_file}")

    # Plot parameter statistics
    plot_parameter_statistics(results, output_dir, style=args.style)

    # Plot AUROC curves if available
    if auroc_results:
        all_datasets = set()
        for auroc_dict in auroc_results.values():
            all_datasets.update(auroc_dict.keys())
        datasets = sorted(all_datasets)

        if datasets:
            # Save AUROC results to CSV before plotting
            csv_file = output_dir / 'auroc_results.csv'
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header: k + all dataset names
                header = ['k'] + datasets
                writer.writerow(header)

                # Data rows
                k_values = sorted([k for k in auroc_results.keys() if k != 'ALL'])
                if 'ALL' in auroc_results:
                    k_values.append('ALL')

                for k in k_values:
                    row = [k]
                    for dataset in datasets:
                        auroc = auroc_results[k].get(dataset, '')
                        row.append(auroc)
                    writer.writerow(row)

            print(f"Saved AUROC results to: {csv_file}")

            print(f"\nPlotting AUROC curves for {len(datasets)} datasets")
            plot_topk_curves(auroc_results, datasets, output_dir, style=args.style)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
