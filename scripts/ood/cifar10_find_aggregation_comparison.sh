#!/bin/bash
# Compare all Fisher aggregation methods for CIFAR-10
# Usage: bash scripts/ood/cifar10_find_aggregation_comparison.sh

ROOT_DIR="results/cifar10_resnet18_32x32_base_e100_lr0.1"
ID_DATA="cifar10"
GPU=0

echo "========================================"
echo "FIND Postprocessor Aggregation Methods"
echo "Comparing 7 methods on CIFAR-10"
echo "========================================"
echo ""

# Array of methods to test
methods=(
    "logsumexp"
    "separated"
    "multiscale"
    "normalized"
    "bayesian"
    "dualpath"
    "adaptive_weight"
)

# Test each method
for method in "${methods[@]}"; do
    echo "----------------------------------------"
    echo "Testing method: $method"
    echo "----------------------------------------"

    CUDA_VISIBLE_DEVICES=$GPU python scripts/eval_ood.py \
        --root $ROOT_DIR \
        --postprocessor find \
        --id-data $ID_DATA \
        --save-csv \
        --postprocessor.postprocessor_args.aggregation_method $method

    if [ $? -eq 0 ]; then
        echo "✓ $method completed successfully"
    else
        echo "✗ $method failed"
    fi
    echo ""
done

echo "========================================"
echo "All tests completed!"
echo "Check results in: $ROOT_DIR"
echo "========================================"
