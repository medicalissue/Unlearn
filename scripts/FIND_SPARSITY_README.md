# FIND Sparsity Discovery Visualization

This script analyzes how **top-k parameter selection** affects OOD detection performance in the FIND (Fisher Energy-based Detection) postprocessor.

## Overview

The FIND postprocessor computes OOD scores using Fisher Energy:

```
S(x) = sum_{i=1}^{d} g_i^2 / F_i^p
```

where:
- `g_i`: gradient of loss w.r.t. i-th FC parameter
- `F_i`: Fisher Information for i-th parameter
- `p`: fisher_power

**Sparsity Discovery** explores selecting only the **top-k parameters** (those with smallest Fisher values, i.e., largest `1/F^p` contribution) instead of using all parameters.

## Key Features

- **Log-scale k sweep**: Tests k values from 1 to total FC parameters using `np.logspace`
- **Automatic parameter detection**: Automatically determines FC layer size (e.g., ~102K for ResNet18/ImageNet-200, ~2M for ResNet50/ImageNet-1K)
- **CSV caching**: Saves results to CSV for easy replotting and reuse
- **Performance visualization**: Generates AUROC vs. k plots with best k highlighted

## Usage

### Basic Usage

```bash
python scripts/visualize_find_sparsity.py \
    --root results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --id-data imagenet200 \
    --num-points 20 \
    --output figures/find_sparsity_analysis.pdf
```

### Advanced Options

```bash
python scripts/visualize_find_sparsity.py \
    --root results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --id-data imagenet200 \
    --num-points 30 \
    --k-min 10 \
    --k-max 100000 \
    --fisher-power 1.0 \
    --gradient-type entropy \
    --seed 0 \
    --output figures/find_sparsity_custom.pdf \
    --csv results/custom_sparsity_results.csv
```

### Load from Existing CSV (Fast Replotting)

```bash
python scripts/visualize_find_sparsity.py \
    --root results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --id-data imagenet200 \
    --load-csv results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/analysis/sparsity_results.csv \
    --output figures/find_sparsity_replot.pdf
```

## Arguments

### Required
- `--root`: Path to experiment directory (contains `s0/best.ckpt`)
- `--id-data`: ID dataset name (e.g., `imagenet200`, `cifar10`)

### Sparsity Sweep
- `--num-points`: Number of k values to test (default: 20)
- `--k-min`: Minimum k value (default: 1)
- `--k-max`: Maximum k value (default: total FC params)

### FIND Parameters
- `--fisher-power`: Fisher power parameter (default: 1.0)
- `--gradient-type`: Gradient type - `nll`, `entropy`, `focal`, `label_smoothing`, `kl` (default: `entropy`)

### Output
- `--output`: Output PDF file path (default: `figures/find_sparsity_analysis.pdf`)
- `--csv`: CSV file to save results (default: auto-generated in `{root}/analysis/`)
- `--load-csv`: Load from existing CSV instead of recomputing

### Other
- `--seed`: Checkpoint seed (default: 0, i.e., `s0/`)
- `--ood-data`: Specific OOD dataset to test (default: all available)

## Output Files

### 1. CSV Results (`{root}/analysis/find_sparsity_results.csv`)

Example:
```csv
topk,auroc,auroc_ninco,auroc_ssb_hard
1,0.723,0.698,0.748
10,0.756,0.731,0.781
100,0.782,0.759,0.805
...
```

Columns:
- `topk`: Number of top-k parameters selected
- `auroc`: Average AUROC across all OOD datasets
- `auroc_{ood_name}`: AUROC for specific OOD dataset

### 2. Visualization (`figures/find_sparsity_analysis.pdf`)

Plot shows:
- **X-axis**: Top-k parameters (log scale)
- **Y-axis**: Average AUROC (%)
- **Red star**: Best k value and corresponding AUROC
- **Info box**: Dataset, Fisher power, gradient type

## Example Workflow

### Step 1: Run Initial Analysis

```bash
# Run with 20 log-spaced points
python scripts/visualize_find_sparsity.py \
    --root results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --id-data imagenet200 \
    --num-points 20 \
    --output figures/find_sparsity_initial.pdf
```

This will:
1. Load checkpoint from `results/.../s0/best.ckpt`
2. Detect FC layer size (e.g., 102,600 params for ResNet18/ImageNet-200)
3. Generate 20 log-spaced k values: [1, ..., 102600]
4. For each k:
   - Set `postprocessor.topk = k`
   - Run OOD evaluation
   - Record AUROC
5. Save results to `results/.../analysis/find_sparsity_results.csv`
6. Plot AUROC vs. k and save to `figures/find_sparsity_initial.pdf`

### Step 2: Refine Search (Optional)

If you want to zoom into a specific k range:

```bash
python scripts/visualize_find_sparsity.py \
    --root results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --id-data imagenet200 \
    --num-points 30 \
    --k-min 1000 \
    --k-max 50000 \
    --output figures/find_sparsity_refined.pdf \
    --csv results/.../analysis/find_sparsity_refined.csv
```

### Step 3: Replot from CSV (No Recomputation)

```bash
python scripts/visualize_find_sparsity.py \
    --root results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --id-data imagenet200 \
    --load-csv results/.../analysis/find_sparsity_results.csv \
    --output figures/find_sparsity_final.pdf
```

## Expected Results

### Typical Parameter Counts

| Dataset | Architecture | Num Classes | Feature Dim | FC Params |
|---------|-------------|-------------|-------------|-----------|
| CIFAR-10 | ResNet18 | 10 | 512 | 5,130 |
| CIFAR-100 | ResNet18 | 100 | 512 | 51,300 |
| ImageNet-200 | ResNet18 | 200 | 512 | 102,600 |
| ImageNet-1K | ResNet50 | 1000 | 2048 | 2,049,000 |

### Hypothesized Behavior

1. **Very low k (k < 100)**: Poor performance (too sparse, missing important parameters)
2. **Optimal k (k ≈ 1K-100K)**: Best AUROC (selects most informative parameters)
3. **High k (k → total)**: Slightly degraded performance (includes noisy parameters)

The visualization will reveal the optimal sparsity level for your specific model and dataset.

## Tips

- **Start with default settings** (`--num-points 20`) to get a rough sense of the k-AUROC curve
- **Use CSV caching** to avoid recomputing scores when adjusting plot aesthetics
- **Test different gradient types** (`--gradient-type`) to see if optimal k varies
- **Compare across datasets** by running on multiple ID datasets (CIFAR-10, ImageNet-200, etc.)

## Troubleshooting

### "Checkpoint not found"
Ensure the checkpoint path is correct:
```bash
ls results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt
```

### "Cannot find FC layer"
The script automatically detects FC layers with common names (`fc`, `head`, `classifier`). If your model uses a different naming convention, you may need to modify `get_fc_params_count()`.

### Out of Memory
Reduce batch size in the script or test fewer k points:
```bash
--num-points 10
```

### Slow Evaluation
Use `--load-csv` to replot without recomputing:
```bash
--load-csv results/.../analysis/find_sparsity_results.csv
```

## Citation

If you use this analysis in your research, please cite the OpenOOD paper:

```bibtex
@article{yang2022openood,
  title={OpenOOD: Benchmarking Generalized Out-of-Distribution Detection},
  author={Yang, Jingkang and others},
  journal={NeurIPS},
  year={2022}
}
```
