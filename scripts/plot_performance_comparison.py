#!/usr/bin/env python
"""
Performance Comparison Scatter Plot
Visualizes OOD detection methods on FPR@95 vs AUROC space
"""

import csv
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
methods = []
avg_data = []
nearood_data = []
with open('/home/junesang/OpenOOD/table.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        methods.append(row[''])
        avg_data.append(row['AVG'])
        nearood_data.append(row['Near-OOD AVG'])

# Parse the data
fpr95_list = []
auroc_list = []
nearood_fpr95_list = []
nearood_auroc_list = []

for data in avg_data:
    fpr, auroc = data.split(' / ')
    fpr95_list.append(float(fpr))
    auroc_list.append(float(auroc))

for data in nearood_data:
    fpr, auroc = data.split(' / ')
    nearood_fpr95_list.append(float(fpr))
    nearood_auroc_list.append(float(auroc))

# Create plot data dictionary
plot_data = {}
for i, method in enumerate(methods):
    plot_data[method] = {
        'FPR@95': fpr95_list[i],
        'AUROC': auroc_list[i],
        'NearOOD_FPR@95': nearood_fpr95_list[i],
        'NearOOD_AUROC': nearood_auroc_list[i]
    }

# Categorize methods
gradnorm_gradrect = ['GradNorm', 'GradRect']
find_methods = ['FInD(p=9)']  # Only p=9
excluded_methods = ['MDS', 'EBO']
other_methods = [m for m in methods if m not in gradnorm_gradrect and m not in find_methods and not m.startswith('FInD') and m not in excluded_methods]

# Create the plot with 2 subplots (vertical layout)
# AUROC range: 75-95 (20 units), FPR range: 40-80 (40 units)
# Ratio: AUROC 5 = FPR 10, so height/width = 20/40 = 0.5
# If width = 12, height = 6 per plot, total height = 12
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# ============ First subplot: AVG ============
# Plot other methods (smaller, lighter)
for method in other_methods:
    data = plot_data[method]
    ax1.scatter(data['FPR@95'], data['AUROC'], s=100, c='gray', alpha=0.4,
               edgecolors='none', zorder=1)
    # Position adjustments for specific methods
    if method == 'ASH':
        ha, va = 'right', 'top'
    elif method == 'ReAct':
        ha, va = 'left', 'top'
    elif method == 'SCALE':
        ha, va = 'left', 'top'
    elif method == 'DICE':
        ha, va = 'right', 'bottom'
    else:
        ha, va = 'left', 'bottom'

    ax1.text(data['FPR@95'], data['AUROC'],
            method, fontsize=14, alpha=0.5, ha=ha, va=va)

# Plot FInD methods (same color, emphasized)
find_colors = '#D62828'  # Vibrant Red
for method in find_methods:
    data = plot_data[method]
    ax1.scatter(data['FPR@95'], data['AUROC'], s=150, c=find_colors, alpha=0.9,
               edgecolors='black', linewidth=1.5, zorder=3, marker='o')
    ax1.text(data['FPR@95'] + 0.2, data['AUROC'] + 0.2,
            'FInD', fontsize=20, fontweight='bold', ha='left', va='bottom',
            color=find_colors)

# Plot GradNorm and GradRect (emphasized, different colors)
colors_grad = {'GradNorm': '#2A9D8F', 'GradRect': '#E76F51'}  # Teal and Burnt Orange
for method in gradnorm_gradrect:
    data = plot_data[method]
    ax1.scatter(data['FPR@95'], data['AUROC'], s=200, c=colors_grad[method], alpha=0.9,
               edgecolors='black', linewidth=1.5, zorder=4, marker='o')
    if method == 'GradNorm':
        ax1.text(data['FPR@95'] + 0.2, data['AUROC'] + 0.2,
                method, fontsize=20, fontweight='bold', ha='left', va='bottom',
                color=colors_grad[method])
    else:  # GradRect
        ax1.text(data['FPR@95'] - 0.2, data['AUROC'] - 0.2,
                method, fontsize=20, fontweight='bold', ha='right', va='top',
                color=colors_grad[method])

# Formatting for first plot
ax1.set_xlabel('FPR@95 (%)', fontsize=16)
ax1.set_ylabel('AUROC (%)', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_title('Overall Average', fontsize=18, fontweight='bold', pad=20)

# Set reasonable axis limits
ax1.set_xlim(40, 85)
ax1.set_ylim(75, 90)

# Set integer ticks
ax1.set_xticks(range(40, 86, 10))
ax1.set_yticks(range(75, 91, 5))

# Set aspect ratio
ax1.set_aspect(2, adjustable='box')

# ============ Second subplot: Near-OOD ============
# Plot other methods
for method in other_methods:
    data = plot_data[method]
    ax2.scatter(data['NearOOD_FPR@95'], data['NearOOD_AUROC'], s=100, c='gray', alpha=0.4,
               edgecolors='none', zorder=1)
    # Position adjustments for specific methods
    if method == 'ASH':
        ha, va = 'right', 'top'
    elif method == 'ReAct':
        ha, va = 'left', 'top'
    elif method == 'SCALE':
        ha, va = 'left', 'top'
    elif method in ['GEN', 'VIM']:
        ha, va = 'right', 'bottom'
    elif method == 'DICE':
        ha, va = 'right', 'bottom'
    else:
        ha, va = 'left', 'bottom'

    ax2.text(data['NearOOD_FPR@95'], data['NearOOD_AUROC'],
            method, fontsize=14, alpha=0.5, ha=ha, va=va)

# Plot FInD methods
for method in find_methods:
    data = plot_data[method]
    ax2.scatter(data['NearOOD_FPR@95'], data['NearOOD_AUROC'], s=150, c=find_colors, alpha=0.9,
               edgecolors='black', linewidth=1.5, zorder=3, marker='o')
    ax2.text(data['NearOOD_FPR@95'] + 0.2, data['NearOOD_AUROC'] + 0.2,
            'FInD', fontsize=20, fontweight='bold', ha='left', va='bottom',
            color=find_colors)

# Plot GradNorm and GradRect
for method in gradnorm_gradrect:
    data = plot_data[method]
    ax2.scatter(data['NearOOD_FPR@95'], data['NearOOD_AUROC'], s=200, c=colors_grad[method], alpha=0.9,
               edgecolors='black', linewidth=1.5, zorder=4, marker='o')
    if method == 'GradNorm':
        ax2.text(data['NearOOD_FPR@95'] + 0.2, data['NearOOD_AUROC'] + 0.2,
                method, fontsize=20, fontweight='bold', ha='left', va='bottom',
                color=colors_grad[method])
    else:  # GradRect
        ax2.text(data['NearOOD_FPR@95'] - 0.2, data['NearOOD_AUROC'] - 0.2,
                method, fontsize=20, fontweight='bold', ha='right', va='top',
                color=colors_grad[method])

# Formatting for second plot
ax2.set_xlabel('FPR@95 (%)', fontsize=16)
ax2.set_ylabel('AUROC (%)', fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_title('Near-OOD Average', fontsize=18, fontweight='bold', pad=20)

# Set reasonable axis limits
ax2.set_xlim(62, 90)
ax2.set_ylim(70, 83)

# Set integer ticks
ax2.set_xticks(range(62, 91, 10))
ax2.set_yticks(range(70, 84, 5))

# Set aspect ratio to match ax1's physical size
# ax1: X=45 units, Y=15 units, aspect=2 → physical ratio = 2 * (15/45) = 0.667
# ax2: X=28 units, Y=13 units → need aspect such that aspect * (13/28) = 0.667
# aspect = 0.667 / (13/28) = 0.667 / 0.464 = 1.437 = 28/13 * 0.667 ≈ 1.437
ax2.set_aspect(28/13 * 2 * (15/45), adjustable='box')

plt.tight_layout()

# Save figure
output_path = 'figures/performance_comparison.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")

plt.show()
