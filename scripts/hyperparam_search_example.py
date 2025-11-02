#!/usr/bin/env python3
"""
Example script for hyperparameter search with result logging.

This script shows how to:
1. Run hyperparameter search using OpenOOD's APS mode
2. Log all results with HyperparamLogger
3. Save and analyze the best configuration

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/hyperparam_search_example.py \
        --config configs/datasets/cifar10/cifar10.yml \
        --postprocessor unlearn \
        --save-dir ./hyperparam_results
"""

import argparse
import os
import sys
from pathlib import Path

# Add OpenOOD to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from openood.postprocessors.hyperparam_logger import HyperparamLogger
from openood.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter Search with Logging')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--postprocessor', type=str, default='unlearn',
                        help='Postprocessor name (e.g., unlearn, odin)')
    parser.add_argument('--save-dir', type=str, default='./hyperparam_results',
                        help='Directory to save results')
    parser.add_argument('--ood-dataset', type=str, default=None,
                        help='Specific OOD dataset to test (optional)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id to use')
    return parser.parse_args()


def run_hyperparameter_search(config_path, postprocessor_name, save_dir, ood_dataset=None, gpu=0):
    """Run hyperparameter search and log results

    Args:
        config_path: Path to dataset config
        postprocessor_name: Name of postprocessor (e.g., 'unlearn')
        save_dir: Directory to save results
        ood_dataset: Specific OOD dataset name (optional)
        gpu: GPU id
    """
    from omegaconf import OmegaConf
    from openood.evaluation_api import Evaluator
    from openood.networks import get_network
    from openood.datasets import get_dataloader

    # Load config
    config = OmegaConf.load(config_path)

    # Setup
    torch.cuda.set_device(gpu)
    logger = setup_logger(name=f'hyperparam_search_{postprocessor_name}')

    # Initialize hyperparameter logger
    hyperparam_logger = HyperparamLogger(
        save_dir=save_dir,
        method_name=postprocessor_name
    )

    # Load postprocessor config
    postprocessor_config_path = f'configs/postprocessors/{postprocessor_name}.yml'
    postprocessor_config = OmegaConf.load(postprocessor_config_path)

    # Merge configs
    config.postprocessor = postprocessor_config.postprocessor

    # Enable APS mode for automatic search
    config.postprocessor.APS_mode = True

    logger.info(f"Running hyperparameter search for {postprocessor_name}")
    logger.info(f"Dataset: {config.dataset.name}")
    logger.info(f"Results will be saved to: {save_dir}")

    # Get sweep parameters
    sweep_config = config.postprocessor.postprocessor_sweep

    # Generate all hyperparameter combinations
    from itertools import product

    # Example: Extract sweep parameters
    param_names = []
    param_values = []

    for key, values in sweep_config.items():
        if isinstance(values, list):
            param_names.append(key)
            param_values.append(values)
        elif isinstance(values, dict):
            # For nested configs like weights
            for sub_key, sub_values in values.items():
                if isinstance(sub_values, list):
                    param_names.append(f"{key}.{sub_key}")
                    param_values.append(sub_values)

    # Generate all combinations
    total_combinations = 1
    for vals in param_values:
        total_combinations *= len(vals)

    logger.info(f"Total hyperparameter combinations to test: {total_combinations}")

    if total_combinations > 1000:
        logger.warning(f"Large number of combinations ({total_combinations}). Consider reducing sweep range.")

    # Load network
    net = get_network(config.network)

    # Run evaluation for each combination
    experiment_count = 0
    for combination in product(*param_values):
        experiment_count += 1

        # Create hyperparameter dict
        hyperparams = dict(zip(param_names, combination))

        logger.info(f"\n{'='*80}")
        logger.info(f"Experiment {experiment_count}/{total_combinations}")
        logger.info(f"Hyperparameters: {hyperparams}")
        logger.info(f"{'='*80}\n")

        # Update config with current hyperparameters
        current_config = config.copy()
        for key, value in hyperparams.items():
            if '.' in key:
                # Nested parameter
                parts = key.split('.')
                setattr(getattr(current_config.postprocessor.postprocessor_args, parts[0]), parts[1], value)
            else:
                # Top-level parameter
                setattr(current_config.postprocessor.postprocessor_args, key, value)

        try:
            # Create evaluator
            evaluator = Evaluator(current_config)

            # Run evaluation
            if ood_dataset:
                # Test on specific OOD dataset
                metrics = evaluator.eval_ood(fsood=False, csood=False, nearood=False)
                # Extract metrics for the specific dataset
                # This is simplified - adjust based on actual Evaluator API
                result_metrics = metrics.get(ood_dataset, {})
            else:
                # Test on all OOD datasets
                metrics = evaluator.eval_ood()
                # Average metrics across all OOD datasets
                result_metrics = {
                    'AUROC': sum(m.get('AUROC', 0) for m in metrics.values()) / len(metrics),
                    'FPR95': sum(m.get('FPR95', 0) for m in metrics.values()) / len(metrics),
                }

            # Log result
            hyperparam_logger.log_result(
                hyperparams=hyperparams,
                metrics=result_metrics,
                dataset_name=config.dataset.name,
                ood_dataset_name=ood_dataset or 'average'
            )

            logger.info(f"Results: {result_metrics}")

        except Exception as e:
            logger.error(f"Error in experiment {experiment_count}: {e}")
            continue

    # Save summary
    hyperparam_logger.save_summary()

    logger.info("Hyperparameter search completed!")


if __name__ == '__main__':
    args = parse_args()
    run_hyperparameter_search(
        config_path=args.config,
        postprocessor_name=args.postprocessor,
        save_dir=args.save_dir,
        ood_dataset=args.ood_dataset,
        gpu=args.gpu
    )
