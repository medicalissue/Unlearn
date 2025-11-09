#!/usr/bin/env python
"""
Visualize FIND Postprocessor Sparsity Discovery Analysis
---------------------------------------------------------
Analyzes how top-k parameter selection affects OOD detection performance.

Usage:
    python scripts/visualize_find_sparsity.py \
        --root results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
        --id-data imagenet200 \
        --postprocessor find \
        --num-points 20 \
        --output figures/find_sparsity_analysis.pdf
"""

import os
import sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(ROOT_DIR)

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32, ResNet18_224x224, ResNet50


def parse_args():
    parser = argparse.ArgumentParser(description='FIND Sparsity Discovery Visualization')

    # Required arguments
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory of the experiment (e.g., results/imagenet200_resnet18_...)')
    parser.add_argument('--id-data', type=str, required=True,
                        choices=['cifar10', 'cifar100', 'imagenet200'],
                        help='ID dataset name')
    parser.add_argument('--postprocessor', type=str, default='find',
                        help='Postprocessor name (default: find)')

    # Sparsity sweep parameters
    parser.add_argument('--num-points', type=int, default=20,
                        help='Number of k values to test (log-spaced)')
    parser.add_argument('--k-min', type=int, default=None,
                        help='Minimum k value (default: 1)')
    parser.add_argument('--k-max', type=int, default=None,
                        help='Maximum k value (default: total FC params)')

    # Output
    parser.add_argument('--output', type=str, default='figures/find_sparsity_extreme_flat.pdf',
                        help='Output PDF file path')
    parser.add_argument('--csv', type=str, default=None,
                        help='CSV file path to save results (default: auto-generate)')
    parser.add_argument('--load-csv', type=str, default=None,
                        help='Load results from existing CSV file instead of re-computing')

    # Other
    parser.add_argument('--batch-size', type=int, default=200,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of data loading workers')
    parser.add_argument('--fsood', action='store_true',
                        help='Enable full-spectrum OOD evaluation')

    return parser.parse_args()


def get_fc_params_count(net):
    """Get total number of parameters in the final FC layer."""
    # Try common attribute names
    fc = None
    for attr_name in ['fc', 'head', 'heads', 'classifier']:
        if hasattr(net, attr_name):
            layer = getattr(net, attr_name)
            if isinstance(layer, nn.Linear):
                fc = layer
                break
            elif isinstance(layer, (nn.Sequential, nn.ModuleList)):
                for module in reversed(list(layer.modules())):
                    if isinstance(module, nn.Linear):
                        fc = module
                        break
                if fc is not None:
                    break

    if fc is None:
        # Search all modules for the last Linear layer
        linear_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
        if linear_layers:
            fc = linear_layers[-1]
        else:
            raise ValueError("Cannot find FC layer in the network")

    # Count parameters: W (out_features × in_features) + b (out_features)
    num_params = fc.weight.numel()
    if fc.bias is not None:
        num_params += fc.bias.numel()

    return num_params, fc.weight.shape[0], fc.weight.shape[1]


def main():
    args = parse_args()

    NUM_CLASSES = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200}
    MODEL = {
        'cifar10': ResNet18_32x32,
        'cifar100': ResNet18_32x32,
        'imagenet200': ResNet18_224x224,
    }

    try:
        num_classes = NUM_CLASSES[args.id_data]
        model_arch = MODEL[args.id_data]
    except KeyError:
        raise NotImplementedError(f'ID dataset {args.id_data} is not supported.')

    # Auto-generate CSV path if not specified
    if args.csv is None:
        csv_dir = os.path.join(args.root, 'analysis')
        os.makedirs(csv_dir, exist_ok=True)
        args.csv = os.path.join(csv_dir, 'find_sparsity_results.csv')

    # Check if we should load from existing CSV
    if args.load_csv and os.path.exists(args.load_csv):
        print(f"Loading results from {args.load_csv}")
        df = pd.read_csv(args.load_csv)
        k_values = df['topk'].values
        auroc_values = df['overall_auroc'].values
        nearood_auroc_values = df['nearood_auroc'].values
        farood_auroc_values = df['farood_auroc'].values

    else:
        # Use s0 checkpoint
        checkpoint_path = os.path.join(args.root, 's0', 'best.ckpt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")

        # Load network
        net = model_arch(num_classes=num_classes)
        net.load_state_dict(torch.load(checkpoint_path))
        net.cuda()
        net.eval()

        # Get FC layer parameter count
        total_params, num_cls, feature_dim = get_fc_params_count(net)
        print(f"\nFC Layer Info:")
        print(f"  Num classes: {num_cls}")
        print(f"  Feature dim: {feature_dim}")
        print(f"  Total FC params: {total_params:,}")

        # Use specific k values: 1, 10, 100, 1000, 10000, and total params
        k_values = np.array([1, 10, 100, 1000, 10000, total_params])

        print(f"\nSparsity Sweep:")
        print(f"  k values to test: {len(k_values)} points")
        print(f"  Values: {k_values.tolist()}")

        # Run evaluations for different k values
        auroc_values = []
        all_metrics = []

        print("\nRunning sparsity sweep...")
        pbar = tqdm(k_values, desc="Testing k values")
        for k in pbar:
            # Create evaluator with specific topk
            evaluator = Evaluator(
                net,
                id_name=args.id_data,
                data_root=os.path.join(ROOT_DIR, 'data'),
                config_root=None,
                preprocessor=None,
                postprocessor_name=args.postprocessor,
                postprocessor=None,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers
            )

            # Override topk in postprocessor
            evaluator.postprocessor.topk = int(k)
            evaluator.postprocessor.use_topk = True

            # Dictionary to store gradient stats for each dataset
            dataset_gradient_stats = {}

            # Process each OOD dataset separately to manage memory efficiently
            for ood_split in ['near', 'far']:
                for dataset_name in evaluator.dataloader_dict['ood'][ood_split].keys():
                    # Enable gradient statistics collection for this dataset only
                    evaluator.postprocessor.collect_gradient_stats = True
                    evaluator.postprocessor.gradient_stats = []

                    # Run inference on this dataset
                    ood_dl = evaluator.dataloader_dict['ood'][ood_split][dataset_name]
                    print(f'Performing inference on {dataset_name} dataset...', flush=True)
                    _, _, _ = evaluator.postprocessor.inference(evaluator.net, ood_dl, progress=True)

                    # Process gradient stats immediately after inference
                    if hasattr(evaluator.postprocessor, 'gradient_stats') and evaluator.postprocessor.gradient_stats:
                        # Combine boolean masks across all batches using logical OR (on CPU)
                        combined_mask = torch.zeros(total_params, dtype=torch.bool, device='cpu')
                        all_avg_gradients = []

                        for stats in evaluator.postprocessor.gradient_stats:
                            combined_mask |= stats['selected_mask']  # Boolean indexing auto-deduplicates
                            all_avg_gradients.append(stats['avg_gradient'])

                        # Store stats for this dataset
                        selected_count = int(combined_mask.sum().item())
                        selected_pct = (combined_mask.sum().item() / total_params) * 100.0
                        avg_grad = np.mean(all_avg_gradients) if all_avg_gradients else 0.0

                        dataset_gradient_stats[dataset_name] = {
                            'selected_count': selected_count,
                            'selected_percentage': selected_pct,
                            'avg_gradient': avg_grad
                        }

                        # Print dataset stats immediately
                        print(f"  → {dataset_name}: {selected_count:,} params ({selected_pct:.2f}%), avg_grad={avg_grad:.4f}")

                        # Free memory
                        del combined_mask

                    # Clear stats for next dataset
                    evaluator.postprocessor.gradient_stats = []
                    evaluator.postprocessor.collect_gradient_stats = False

            # Now run the full evaluation to get metrics (without gradient stats collection)
            metrics = evaluator.eval_ood(fsood=args.fsood, progress=True)

            # Calculate overall statistics (average across all datasets)
            # For overall, we compute the average of per-dataset statistics
            if dataset_gradient_stats:
                all_selected_counts = [stats['selected_count'] for stats in dataset_gradient_stats.values()]
                all_selected_percentages = [stats['selected_percentage'] for stats in dataset_gradient_stats.values()]
                all_gradients = [stats['avg_gradient'] for stats in dataset_gradient_stats.values()]

                gradient_stats_per_dataset = {
                    'selected_count': np.mean(all_selected_counts),
                    'selected_percentage': np.mean(all_selected_percentages),
                    'avg_per_sample': k,  # For top-k, each sample selects exactly k params
                    'avg_per_sample_percentage': (k / total_params) * 100.0,
                    'avg_gradient_mean': np.mean(all_gradients),
                    'avg_gradient_std': np.std(all_gradients),
                    # Add per-dataset stats
                    'per_dataset': dataset_gradient_stats
                }
            else:
                gradient_stats_per_dataset = {
                    'per_dataset': {}
                }

            # Extract AUROC from DataFrame
            # metrics is a pandas DataFrame with datasets as rows
            # Calculate: nearood, farood, and overall average (excluding nearood/farood aggregates)

            # Get individual datasets (exclude aggregate rows)
            individual_datasets = [idx for idx in metrics.index if idx not in ['nearood', 'farood']]

            if individual_datasets:
                # Overall average from individual datasets
                overall_auroc = float(metrics.loc[individual_datasets, 'AUROC'].mean())
                overall_fpr = float(metrics.loc[individual_datasets, 'FPR@95'].mean())
            else:
                overall_auroc = 0.0
                overall_fpr = 0.0

            # Get nearood and farood if they exist
            if 'nearood' in metrics.index:
                nearood_auroc = float(metrics.loc['nearood', 'AUROC'])
                nearood_fpr = float(metrics.loc['nearood', 'FPR@95'])
            else:
                nearood_auroc = overall_auroc
                nearood_fpr = overall_fpr

            if 'farood' in metrics.index:
                farood_auroc = float(metrics.loc['farood', 'AUROC'])
                farood_fpr = float(metrics.loc['farood', 'FPR@95'])
            else:
                farood_auroc = overall_auroc
                farood_fpr = overall_fpr

            # Print result immediately for user feedback
            result_str = f"✓ k={k:,}: "
            result_str += f"AUROC: avg={overall_auroc:.2f}%, near={nearood_auroc:.2f}%, far={farood_auroc:.2f}% | "
            result_str += f"FPR95: avg={overall_fpr:.2f}%, near={nearood_fpr:.2f}%, far={farood_fpr:.2f}%"
            print(f"\n{result_str}")

            # Store values for plotting
            auroc_values.append(overall_auroc)
            all_metrics.append({
                'topk': k,
                'auroc': overall_auroc,
                'overall_fpr': overall_fpr,
                'metrics': metrics,
                'gradient_stats': gradient_stats_per_dataset
            })

        auroc_values = np.array(auroc_values)

        # Save results to CSV
        df = pd.DataFrame({
            'topk': k_values,
            'overall_auroc': auroc_values,
        })

        # Add pre-computed metrics from all_metrics
        if all_metrics:
            # Removed nearood_auroc, farood_auroc, nearood_fpr95, farood_fpr95
            df['overall_fpr95'] = [m['overall_fpr'] for m in all_metrics]

            # Add gradient statistics
            df['selected_params_count'] = [m['gradient_stats'].get('selected_count', k) for m, k in zip(all_metrics, k_values)]
            df['selected_params_percentage'] = [m['gradient_stats'].get('selected_percentage', (k/total_params)*100) for m, k in zip(all_metrics, k_values)]
            df['avg_params_per_sample'] = [m['gradient_stats'].get('avg_per_sample', k) for m, k in zip(all_metrics, k_values)]
            df['avg_params_per_sample_pct'] = [m['gradient_stats'].get('avg_per_sample_percentage', (k/total_params)*100) for m, k in zip(all_metrics, k_values)]
            df['avg_gradient_magnitude'] = [m['gradient_stats'].get('avg_gradient_mean', np.nan) for m in all_metrics]
            df['avg_gradient_magnitude_std'] = [m['gradient_stats'].get('avg_gradient_std', np.nan) for m in all_metrics]

            # Add individual dataset metrics (AUROC and FPR95)
            for dataset in ['ssb_hard', 'ninco', 'inaturalist', 'textures', 'openimage_o']:
                if dataset in all_metrics[0]['metrics'].index:
                    df[f'{dataset}_auroc'] = [float(m['metrics'].loc[dataset, 'AUROC']) for m in all_metrics]
                    df[f'{dataset}_fpr95'] = [float(m['metrics'].loc[dataset, 'FPR@95']) for m in all_metrics]

                    # Add per-dataset gradient statistics
                    df[f'{dataset}_selected_count'] = [
                        m['gradient_stats'].get('per_dataset', {}).get(dataset, {}).get('selected_count', np.nan)
                        for m in all_metrics
                    ]
                    df[f'{dataset}_selected_percentage'] = [
                        m['gradient_stats'].get('per_dataset', {}).get(dataset, {}).get('selected_percentage', np.nan)
                        for m in all_metrics
                    ]
                    df[f'{dataset}_avg_gradient'] = [
                        m['gradient_stats'].get('per_dataset', {}).get(dataset, {}).get('avg_gradient', np.nan)
                        for m in all_metrics
                    ]

        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\n✓ Saved results to {args.csv}")

    # Create visualization
    print(f"\nCreating visualization...")

    # Get FPR95 values from CSV if available
    if 'overall_fpr95' in df.columns:
        fpr_values = df['overall_fpr95'].values
    else:
        fpr_values = None

    # Create figure with two subplots (vertical layout)
    if fpr_values is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    else:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = None

    # ============ AUROC Plot ============
    # Plot with regular scale using a beautiful color palette
    ax1.plot(k_values, auroc_values, 'o-', linewidth=2, markersize=5,
             label='Overall Average', color='#2E86AB')  # Ocean blue

    # Set x-axis to log scale
    ax1.set_xscale('log')

    # Mark best k for overall metric
    best_overall_idx = np.argmax(auroc_values)
    best_overall_k = k_values[best_overall_idx]
    best_overall_auroc = auroc_values[best_overall_idx]

    # Mark best point with subtle donut-style marker
    ax1.plot(best_overall_k, best_overall_auroc, 'o', color='white', markersize=8,
             markeredgecolor='#2E86AB', markeredgewidth=2, zorder=10, alpha=0.7)

    # Formatting
    ax1.set_xlabel('Top-k Parameters', fontsize=20)
    ax1.set_ylabel('AUROC (%)', fontsize=20)
    ax1.set_ylim(70, 90)
    ax1.tick_params(axis='both', which='major', labelsize=18)

    # Format x-axis to show 10^k notation
    from matplotlib.ticker import LogFormatterMathtext
    ax1.xaxis.set_major_formatter(LogFormatterMathtext())

    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=18, loc='lower right')

    # ============ FPR95 Plot ============
    if ax2 is not None and fpr_values is not None:
        # Plot FPR95 (lower is better, so we want to minimize)
        ax2.plot(k_values, fpr_values, 'o-', linewidth=2, markersize=5,
                 label='Overall Average', color='#2E86AB')

        # Set x-axis to log scale
        ax2.set_xscale('log')

        # Mark best k for overall metric (minimum FPR95)
        best_overall_fpr_idx = np.argmin(fpr_values)
        best_overall_fpr_k = k_values[best_overall_fpr_idx]
        best_overall_fpr = fpr_values[best_overall_fpr_idx]

        # Mark best point with subtle donut-style marker
        ax2.plot(best_overall_fpr_k, best_overall_fpr, 'o', color='white', markersize=8,
                 markeredgecolor='#2E86AB', markeredgewidth=2, zorder=10, alpha=0.7)

        # Formatting
        ax2.set_xlabel('Top-k Parameters', fontsize=20)
        ax2.set_ylabel('FPR@95 (%)', fontsize=20)
        ax2.set_ylim(30, 60)
        ax2.tick_params(axis='both', which='major', labelsize=18)

        # Format x-axis to show 10^k notation
        ax2.xaxis.set_major_formatter(LogFormatterMathtext())

        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(fontsize=18, loc='lower right')

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved figure to {args.output}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SPARSITY DISCOVERY RESULTS")
    print(f"{'='*60}")
    print(f"\nBest Overall Average:")
    print(f"  k={best_overall_k:,} ({best_overall_k/k_values[-1]*100:.2f}% of total params)")
    print(f"  AUROC={best_overall_auroc:.2f}%")
    print(f"\nAUROC range: [{auroc_values.min():.2f}%, {auroc_values.max():.2f}%]")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
