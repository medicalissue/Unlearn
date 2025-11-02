# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenOOD is a comprehensive benchmarking framework for Out-of-Distribution (OOD) detection, supporting 60+ methods across anomaly detection, open set recognition, and OOD detection tasks. The framework emphasizes reproducibility and fair comparison across methods originally developed for different detection paradigms.

## Installation and Setup

```bash
# Install from source
pip install git+https://github.com/Jingkang50/OpenOOD
pip install libmr

# For CLIP support
pip install git+https://github.com/openai/CLIP.git
```

## Environment Requirements

**CRITICAL**: Before running any code in this repository, verify the following environment conditions are met:

### Required Environment
1. **Docker Container**: Must be running inside the `junesang` docker container
2. **Conda Environment**: Must activate the `openoodfin` conda environment

### Environment Verification

Always check the environment before executing code:

```bash
# Check conda environment
echo $CONDA_DEFAULT_ENV
# Expected output: openoodfin

# Check Python path (should be in openoodfin environment)
which python
# Expected output: /home/junesang/.conda/envs/.../openoodfin/bin/python

# Verify Python version
python --version
# Expected output: Python 3.10.8
```

### Environment Check Script

Before running any Python commands, use this verification:

```bash
# Quick environment check
if [ "$CONDA_DEFAULT_ENV" = "openoodfin" ]; then
    echo "✓ Correct conda environment: openoodfin"
else
    echo "✗ Wrong conda environment: $CONDA_DEFAULT_ENV"
    echo "  Please activate: conda activate openoodfin"
    exit 1
fi
```

### Activating the Environment

If not in the correct environment:

```bash
# Activate the openoodfin conda environment
conda activate openoodfin

# Verify activation
echo $CONDA_DEFAULT_ENV
```

### Important Notes
- **DO NOT** run code outside the `openoodfin` conda environment
- **DO NOT** run code outside the `junesang` docker container
- Always verify environment before executing training or evaluation scripts
- The working directory should be `/home/junesang/OpenOOD`

## Common Commands

### Training

```bash
# Basic training with config files
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/base_preprocessor.yml

# Training with method-specific configs
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/train_mos.yml \
    configs/preprocessors/base_preprocessor.yml \
    --optimizer.num_epochs 100
```

### Evaluation

```bash
# Standard OOD evaluation
python scripts/eval_ood.py \
    --root results/cifar10_resnet18_32x32_base_e100_lr0.1 \
    --postprocessor msp \
    --id-data cifar10 \
    --save-csv

# Full-spectrum OOD evaluation
python scripts/eval_ood.py \
    --root results/cifar10_resnet18_32x32_base_e100_lr0.1 \
    --postprocessor msp \
    --id-data cifar10 \
    --fsood

# ImageNet evaluation
python scripts/eval_ood_imagenet.py \
    --root results/imagenet_resnet50_base \
    --postprocessor msp \
    --id-data imagenet
```

### Hyperparameter Search

The framework supports automatic hyperparameter search (APS mode) for postprocessors:

```bash
# Enable APS mode in config
# Set APS_mode: true in configs/postprocessors/<method>.yml

# The evaluator will automatically search hyperparameters
# based on validation ID/OOD performance
```

## Code Architecture

### High-Level Flow

1. **Config System**: YAML-based configuration files define datasets, networks, pipelines, and postprocessors
2. **Pipeline**: Entry point that orchestrates training or evaluation (`openood/pipelines/`)
3. **Trainer**: Handles model training with method-specific logic (`openood/trainers/`)
4. **Evaluator**: Manages OOD detection evaluation (`openood/evaluation_api/`)
5. **Postprocessor**: Applies post-hoc OOD scoring methods (`openood/postprocessors/`)

### Key Components

#### Pipelines (`openood/pipelines/`)
- `train_pipeline.py`: Standard training flow
- `test_ood_pipeline.py`: OOD evaluation flow
- `test_ood_pipeline_aps.py`: Automatic hyperparameter search
- Method-specific pipelines (e.g., `train_oe_pipeline.py`, `train_opengan_pipeline.py`)

#### Postprocessors (`openood/postprocessors/`)
All postprocessors inherit from `BasePostprocessor` and implement:
- `setup()`: Initialize with ID/OOD validation data
- `postprocess()`: Compute OOD scores for a batch
- `inference()`: Run inference on a dataloader
- `set_hyperparam()` / `get_hyperparam()`: For APS mode

Important implementation notes:
- Use `@torch.no_grad()` for inference methods
- Scores should be higher for OOD samples (convention: negate if needed)
- Support both single-pass and multi-step updates

#### Trainers (`openood/trainers/`)
Method-specific training logic:
- Training-time OOD methods (e.g., `oe_trainer.py`, `mos_trainer.py`)
- Data augmentation methods (e.g., `mixup_trainer.py`, `cutmix_trainer.py`)
- Anomaly detection methods (e.g., `dsvdd_trainer.py`, `draem_trainer.py`)

#### Networks (`openood/networks/`)
Backbone architectures and method-specific wrappers:
- CNN backbones: ResNet-18/50, WideResNet
- Transformer backbones: ViT, Swin
- Method-specific networks (e.g., `godin_net.py`, `cider_net.py`)

### Configuration System

Config files are hierarchical and can be overridden via command line:

```bash
# Override config values
python main.py \
    --config configs/datasets/cifar10/cifar10.yml \
    --dataset.image_size 32 \
    --optimizer.lr 0.01 \
    --network.name resnet18_32x32
```

Config structure:
- `configs/datasets/`: Dataset definitions (ID and OOD splits)
- `configs/networks/`: Network architectures
- `configs/pipelines/`: Training/evaluation workflows
- `configs/postprocessors/`: Post-hoc method configurations
- `configs/preprocessors/`: Data preprocessing settings

### Directory Structure

```
OpenOOD/
├── data/                           # Dataset storage
│   ├── benchmark_imglist/         # Dataset split files
│   ├── images_classic/            # CIFAR, MNIST, etc.
│   └── images_largescale/         # ImageNet variants
├── results/                        # Model checkpoints and outputs
│   └── checkpoints/               # Pre-trained models
│       └── <dataset>_<arch>_<method>_s<seed>/
│           ├── best.ckpt          # Best checkpoint
│           ├── postprocessors/    # Saved postprocessor states
│           └── scores/            # Pre-computed OOD scores
├── openood/
│   ├── datasets/                  # Dataset loaders
│   ├── networks/                  # Network architectures
│   ├── trainers/                  # Training methods
│   ├── postprocessors/            # OOD scoring methods
│   ├── pipelines/                 # Orchestration logic
│   ├── evaluation_api/            # Evaluator class
│   └── utils/                     # Utilities
└── scripts/                       # Execution scripts
    ├── eval_ood.py               # Main evaluation script
    ├── basics/                    # Training scripts by dataset
    ├── ood/                       # Method-specific scripts
    └── download/                  # Data download scripts
```

### Evaluation Flow

1. Load trained model checkpoint from `results/<experiment>/best.ckpt`
2. Initialize network architecture with method-specific wrapper if needed
3. Create `Evaluator` with:
   - Network
   - ID dataset name
   - Postprocessor name/instance
   - Data paths
4. Evaluator automatically:
   - Loads ID test and OOD test datasets
   - Initializes postprocessor with validation data if needed
   - Computes predictions and OOD scores
   - Calculates metrics (AUROC, AUPR, FPR95)
5. Results saved to CSV or printed

### Adding New Methods

#### Post-Hoc Method
1. Create `openood/postprocessors/<method>_postprocessor.py`
2. Inherit from `BasePostprocessor`
3. Implement required methods: `setup()`, `postprocess()`, `inference()`
4. Create config: `configs/postprocessors/<method>.yml`
5. Add to `openood/postprocessors/__init__.py`

#### Training Method
1. Create `openood/trainers/<method>_trainer.py`
2. Inherit from `BaseTrainer`
3. Implement `train_epoch()` and optionally `eval_epoch()`
4. Create network wrapper if needed in `openood/networks/`
5. Create pipeline config: `configs/pipelines/train/train_<method>.yml`

## Current Work: Unlearn Method

The repository includes an "unlearn" postprocessor (`openood/postprocessors/unlearn_postprocessor.py`) that performs gradient-based test-time adaptation:

### Key Concepts
- **Unlearning modes**: `ascent` (gradient ascent) or `fisher` (Fisher-weighted updates)
- **FC-only updates**: Only updates final classification layer for efficiency
- **Batch parallelization**: Uses `torch.func.vmap` for efficient batched processing
- **Multiple metrics**: Supports energy, confidence, margin, entropy, KL divergence, etc.

### Hyperparameters
- `eta`: Learning rate for gradient updates
- `num_steps`: Number of unlearning iterations
- `temp`: Temperature for soft pseudo-labels
- `unlearn_mode`: "ascent" or "fisher"
- `score_type`: Which metric to use ("delta_energy", "combo", etc.)
- `weights`: Weight coefficients for combo scoring

### Implementation Notes
- Uses `torch.func.grad` for gradient computation (vmap-compatible)
- Avoids in-place operations and conditional logic for vmap compatibility
- Normalizes logits for numerical stability
- Returns higher scores for OOD samples

## Testing

```bash
# Run ID accuracy test
python scripts/basics/cifar10/test_cifar10.py \
    --config configs/datasets/cifar10/cifar10.yml \
    configs/networks/resnet18_32x32.yml \
    --checkpoint results/.../best.ckpt

# Run OOD evaluation with multiple seeds
# The eval_ood.py script automatically iterates over s0, s1, s2, etc.
python scripts/eval_ood.py --root results/experiment_name --postprocessor msp
```

## Pre-commit Hooks

The repository uses pre-commit hooks for code quality:

```bash
# Install pre-commit
pip install -U pre-commit
pre-commit install

# Manually run checks
pre-commit run --all-files
```

Checks include: flake8, yapf, isort, trailing whitespace, markdown linting.

## Important Notes

- **Checkpoint structure**: Training runs create subdirectories `s0/`, `s1/`, `s2/` for different seeds
- **Postprocessor caching**: Postprocessors with setup can be saved to `postprocessors/<name>.pkl` for reuse
- **Score caching**: Pre-computed scores can be saved to `scores/<name>.pkl` to avoid recomputation
- **APS mode**: When enabled, postprocessors search hyperparameters using validation sets
- **ID datasets**: Primary focus on CIFAR-10, CIFAR-100, ImageNet-200, ImageNet-1K
- **Score convention**: Higher scores indicate OOD (some methods like MSP need negation)
- **Feature extraction**: Networks should support `return_feature=True` for feature-based methods

## Debugging Tips

- Check config file paths are correct (relative to project root)
- Verify checkpoint directory structure matches expected `s*/best.ckpt` pattern
- For custom networks, ensure they support the expected interface (e.g., `return_feature`)
- Use `--save-score` flag to cache scores for faster re-evaluation with different metrics
- Enable progress bars by default (set `progress=True` in inference)
