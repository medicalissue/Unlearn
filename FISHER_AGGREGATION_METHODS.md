# Fisher Aggregation Methods for FIND Postprocessor

## Overview

The FIND postprocessor now supports **7 different aggregation methods** for combining Fisher-weighted gradient information. This allows you to leverage both small and large Fisher values in different ways.

## Quick Start

To use a different aggregation method, simply change the `aggregation_method` parameter in `configs/postprocessors/find.yml`:

```yaml
postprocessor_args:
  aggregation_method: dualpath  # Change this to any method below
```

## Available Methods

### 1. **logsumexp** (Original/Baseline)

**Formula**: `S(x) = log(Σ exp(log(g²/F^p)))`

**Characteristics**:
- Original FIND method
- Dominated by small Fisher values (flat directions)
- LogSumExp makes large F dimensions nearly invisible

**When to use**: Baseline comparison

**Config**:
```yaml
aggregation_method: logsumexp
fisher_power: 1.0
```

---

### 2. **separated** (Small-F vs Large-F Split)

**Formula**:
- Small F: `S_small = mean(log(g²/F^p))` for F < threshold
- Large F: `S_large = mean(log(g²*F^q))` for F ≥ threshold
- Total: `S(x) = w_small * S_small + w_large * S_large`

**Characteristics**:
- Explicitly separates small and large Fisher dimensions
- Uses **inverse Fisher** (g²/F^p) for flat directions
- Uses **forward Fisher** (g²*F^q) for curvature directions
- Most direct implementation of dual information usage

**When to use**: When you want explicit control over small/large F contributions

**Config**:
```yaml
aggregation_method: separated
sep_threshold_percentile: 50.0  # Split at median
sep_w_small: 1.0  # Weight for small-F score
sep_w_large: 1.0  # Weight for large-F score
sep_large_power: 1.0  # Power for large-F dimensions
```

**Recommended starting points**:
- Balanced: `sep_w_small: 1.0, sep_w_large: 1.0`
- Emphasize flat directions: `sep_w_small: 2.0, sep_w_large: 1.0`
- Emphasize curvature: `sep_w_small: 1.0, sep_w_large: 2.0`

---

### 3. **multiscale** (Percentile-based)

**Formula**: `S(x) = Σ w_i * percentile_i(log(g²/F^p))`

**Characteristics**:
- Uses multiple percentiles of the Fisher energy distribution
- High percentiles (e.g., 90) capture small-F influence
- Low percentiles (e.g., 10) capture large-F influence
- Simple and robust to outliers

**When to use**: Quick experiments without explicit thresholding

**Config**:
```yaml
aggregation_method: multiscale
ms_percentiles: [90, 50, 10]  # High, median, low
ms_weights: [1.0, 1.0, 1.0]  # Equal weights
```

**Recommended variations**:
- Emphasize small-F: `ms_weights: [2.0, 1.0, 0.5]`
- Emphasize large-F: `ms_weights: [0.5, 1.0, 2.0]`
- Focus on extremes: `ms_percentiles: [95, 50, 5]`

---

### 4. **normalized** (Group-wise Normalization)

**Formula**:
- Split dimensions into N groups by Fisher value
- Normalize within each group: `z = (x - μ) / σ`
- Average normalized scores across groups

**Characteristics**:
- Prevents any group from dominating by normalization
- Each Fisher value range contributes equally
- Most balanced approach

**When to use**: When Fisher values span many orders of magnitude

**Config**:
```yaml
aggregation_method: normalized
norm_num_groups: 5  # Number of groups (try 3-10)
```

**Tips**:
- More groups (10) → finer granularity
- Fewer groups (3) → more robust to noise

---

### 5. **bayesian** (Uncertainty + Confidence)

**Formula**:
- Uncertainty: `U(x) = -log(Σ g²/F^p)` (small F dominant)
- Confidence: `C(x) = log(Σ g²*F)` (large F weighted)
- Total: `S(x) = w_u * U(x) + w_c * C(x)`

**Characteristics**:
- Probabilistic interpretation
- Uncertainty captures anomalies in flat directions
- Confidence captures deviations in well-learned directions
- Theoretically motivated

**When to use**: When you want interpretable components

**Config**:
```yaml
aggregation_method: bayesian
bayes_uncertainty_weight: 1.0
bayes_confidence_weight: 1.0
```

**Interpretation**:
- High uncertainty + Low confidence → Far OOD
- Low uncertainty + High confidence → Near OOD
- Both low → In-distribution

---

### 6. **dualpath** (Inverse + Forward Paths)

**Formula**:
- Inverse path: `S_inv = agg(log(g²/F^p))`
- Forward path: `S_fwd = agg(log(g²*F^p))`
- Total: `S(x) = α * S_inv + β * S_fwd`

**Characteristics**:
- Two separate scoring paths with different aggregators
- Can use `median` (robust), `mean` (smooth), or `logsumexp` (peaked)
- Most flexible dual-path approach

**When to use**: When you want robust aggregation with dual signals

**Config**:
```yaml
aggregation_method: dualpath
dual_alpha: 0.5  # Weight for inverse path
dual_beta: 0.5  # Weight for forward path
dual_aggregator: median  # "median", "mean", or "logsumexp"
```

**Recommended combinations**:
- Robust: `dual_aggregator: median, dual_alpha: 0.5, dual_beta: 0.5`
- Smooth: `dual_aggregator: mean, dual_alpha: 0.6, dual_beta: 0.4`
- Sharp: `dual_aggregator: logsumexp, dual_alpha: 0.3, dual_beta: 0.7`

---

### 7. **adaptive_weight** (Sigmoid-based)

**Formula**:
- Normalize: `F_norm = (log(F) - μ) / σ`
- Weights: `w = sigmoid(-F_norm / T)`
- Score: `S(x) = Σ w * log(g²/F^p)`

**Characteristics**:
- Smooth transition between small and large F
- Temperature controls how sharp the weighting is
- No hard threshold needed

**When to use**: When you want smooth, continuous weighting

**Config**:
```yaml
aggregation_method: adaptive_weight
aw_temperature: 1.0  # Higher = smoother weights
```

**Temperature guide**:
- `T = 0.1` → Sharp (almost binary weights)
- `T = 1.0` → Moderate (sigmoid-like)
- `T = 5.0` → Smooth (nearly uniform)

---

## Experimental Guide

### Step 1: Baseline
```yaml
aggregation_method: logsumexp
```
Run evaluation to establish baseline performance.

### Step 2: Try Dual-Path Methods
Start with the most promising methods:

```yaml
# Try separated first (most direct)
aggregation_method: separated
sep_threshold_percentile: 50.0
sep_w_small: 1.0
sep_w_large: 1.0
```

```yaml
# Try dualpath with median (most robust)
aggregation_method: dualpath
dual_alpha: 0.5
dual_beta: 0.5
dual_aggregator: median
```

### Step 3: Fine-tune Weights
Based on Step 2 results, adjust weights:
- If far-OOD detection is weak → increase small-F weight
- If near-OOD detection is weak → increase large-F weight

### Step 4: Try Other Methods
Explore other approaches:

```yaml
# Multiscale (fast to try)
aggregation_method: multiscale
ms_percentiles: [90, 50, 10]
ms_weights: [1.0, 1.0, 1.0]
```

```yaml
# Bayesian (interpretable)
aggregation_method: bayesian
bayes_uncertainty_weight: 1.0
bayes_confidence_weight: 1.0
```

---

## Command Line Override

You can override the aggregation method from the command line:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_ood.py \
    --root results/cifar10_resnet18_32x32_base_e100_lr0.1 \
    --postprocessor find \
    --id-data cifar10 \
    --postprocessor.postprocessor_args.aggregation_method separated \
    --postprocessor.postprocessor_args.sep_w_small 1.5 \
    --postprocessor.postprocessor_args.sep_w_large 1.0
```

---

## Method Comparison

| Method | Small-F Usage | Large-F Usage | Robustness | Flexibility | Speed |
|--------|--------------|---------------|------------|-------------|-------|
| logsumexp | ✓✓✓ | ✗ | Low | Low | Fast |
| separated | ✓✓ | ✓✓ | Medium | High | Fast |
| multiscale | ✓✓ | ✓ | High | Medium | Fast |
| normalized | ✓✓ | ✓✓ | High | Medium | Medium |
| bayesian | ✓✓ | ✓✓ | Medium | High | Fast |
| dualpath | ✓✓ | ✓✓ | High | High | Fast |
| adaptive_weight | ✓✓ | ✓ | Medium | Medium | Fast |

---

## Theory Summary

### Why Small Fisher Values?
- **Low curvature**: Parameters in flat loss directions
- **Poorly constrained**: Model is uncertain about these parameters
- **Sensitive to OOD**: Unusual activations in flat directions indicate OOD

### Why Large Fisher Values?
- **High curvature**: Parameters in steep loss directions
- **Well constrained**: Model is confident about these parameters
- **Reliable signal**: Deviations in well-learned directions indicate OOD

### The Key Insight
Both small and large Fisher values provide complementary information:
- **Small F**: Captures *unusual* behavior in *uncertain* dimensions
- **Large F**: Captures *unexpected* behavior in *confident* dimensions

The original LogSumExp method only uses small-F information. The new methods leverage both.

---

## Next Steps

1. **Quick test**: Try `separated` or `dualpath` with default settings
2. **Analyze results**: Check if far-OOD vs near-OOD performance changes
3. **Tune weights**: Adjust to balance different OOD types
4. **Compare methods**: Run all methods and compare AUROC/FPR95
5. **Iterate**: Fine-tune the best-performing method

---

## Implementation Details

All methods are implemented in [openood/postprocessors/find_postprocessor.py](openood/postprocessors/find_postprocessor.py):

- `_aggregate_logsumexp()`: Original method
- `_aggregate_separated()`: Split small/large F
- `_aggregate_multiscale()`: Percentile-based
- `_aggregate_normalized()`: Group normalization
- `_aggregate_bayesian()`: Uncertainty + confidence
- `_aggregate_dualpath()`: Dual inverse/forward paths
- `_aggregate_adaptive_weight()`: Sigmoid weighting

All methods are numerically stable and GPU-accelerated.
