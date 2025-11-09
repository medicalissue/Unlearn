#!/bin/bash
# Example script to run FIND Sparsity Discovery analysis

# ImageNet-200 example
CUDA_VISIBLE_DEVICES=0 python scripts/visualize_find_sparsity.py \
    --root results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
    --id-data imagenet200 \
    --postprocessor find \
    --num-points 20 \
    --batch-size 200 \
    --num-workers 8 \
    --output figures/find_sparsity_imagenet200.pdf \
    --csv results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/analysis/sparsity_results.csv

# CIFAR-10 example (uncomment to use)
# CUDA_VISIBLE_DEVICES=0 python scripts/visualize_find_sparsity.py \
#     --root results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
#     --id-data cifar10 \
#     --postprocessor find \
#     --num-points 20 \
#     --batch-size 200 \
#     --num-workers 8 \
#     --output figures/find_sparsity_cifar10.pdf

# Load from existing CSV (for quick re-plotting)
# python scripts/visualize_find_sparsity.py \
#     --root results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
#     --id-data imagenet200 \
#     --load-csv results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/analysis/sparsity_results.csv \
#     --output figures/find_sparsity_imagenet200_replot.pdf
