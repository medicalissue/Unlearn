#!/bin/bash
# Quick sweep with fewer k values for faster testing

# Configuration
ARCH="resnet50"
CKPT="/home/junesang/OpenOOD/results/pretrained_weights/resnet50_imagenet1k_v1.pth"
GPU=2
BATCH_SIZE=500
OUTPUT_DIR="./topk_sweep_results_quick"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Top-k values: Carefully selected key points (ResNet50 FC has 2,049,000 params)
# Cover 3 orders of magnitude with ~3 points per decade
TOPK_VALUES=(1 10 100 1000 10000 100000 1000000 -1)

echo "Starting QUICK Top-k sweep experiment..."
echo "Architecture: ${ARCH}"
echo "Checkpoint: ${CKPT}"
echo "Top-k values: ${TOPK_VALUES[@]}"
echo "Total runs: ${#TOPK_VALUES[@]}"
echo ""

for k in "${TOPK_VALUES[@]}"; do
    echo "=========================================="
    echo "Running with top-k = ${k}"
    echo "=========================================="

    # Determine use_topk flag
    if [ "$k" -eq -1 ]; then
        USE_TOPK="False"
        K_VALUE=1000  # Dummy value, won't be used
        SUFFIX="all"
    else
        USE_TOPK="True"
        K_VALUE=$k
        SUFFIX="k${k}"
    fi

    # Update config file temporarily
    CONFIG_FILE="configs/postprocessors/find.yml"
    BACKUP_FILE="${OUTPUT_DIR}/find.yml.backup"

    # Backup original config
    cp ${CONFIG_FILE} ${BACKUP_FILE}

    # Modify config using sed
    sed -i "s/use_topk: .*/use_topk: ${USE_TOPK}/" ${CONFIG_FILE}
    sed -i "s/topk: .*/topk: ${K_VALUE}/" ${CONFIG_FILE}

    # Run evaluation and save output
    OUTPUT_FILE="${OUTPUT_DIR}/topk_${SUFFIX}.log"
    CUDA_VISIBLE_DEVICES=${GPU} python scripts/eval_ood_imagenet.py \
        --arch ${ARCH} \
        --ckpt-path ${CKPT} \
        --postprocessor find \
        --batch-size ${BATCH_SIZE} \
        2>&1 | tee ${OUTPUT_FILE}

    # Restore original config
    cp ${BACKUP_FILE} ${CONFIG_FILE}

    echo ""
    echo "Results saved to: ${OUTPUT_FILE}"
    echo ""

    # Optional: Add delay between runs
    sleep 2
done

echo "=========================================="
echo "QUICK Top-k sweep completed!"
echo "Results directory: ${OUTPUT_DIR}"
echo "=========================================="

# Parse results and create summary (same as full version)
python - <<EOF
import re
import glob

results = {}
log_files = sorted(glob.glob("${OUTPUT_DIR}/topk_*.log"))

for log_file in log_files:
    # Extract k value from filename
    if "topk_all" in log_file:
        k = "ALL"
    else:
        match = re.search(r'topk_k(\d+)', log_file)
        if match:
            k = int(match.group(1))
        else:
            continue

    # Parse AUROC values
    with open(log_file, 'r') as f:
        content = f.read()

    # Find AUROC for different OOD datasets
    auroc_dict = {}

    # Pattern: dataset name followed by AUROC
    for line in content.split('\n'):
        if 'AUROC' in line or 'FPR95' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if 'AUROC' in part and i > 0:
                    dataset = parts[i-1]
                    try:
                        auroc = float(parts[i+1].strip('%,'))
                        auroc_dict[dataset] = auroc
                    except:
                        pass

    if auroc_dict:
        results[k] = auroc_dict

# Print summary table
print("\n" + "="*80)
print("SUMMARY: AUROC vs. Top-k (Quick Sweep)")
print("="*80)

if results:
    # Get all dataset names
    all_datasets = set()
    for auroc_dict in results.values():
        all_datasets.update(auroc_dict.keys())

    all_datasets = sorted(all_datasets)

    # Print header
    print(f"{'Top-k':<12}", end='')
    for dataset in all_datasets:
        print(f"{dataset:<15}", end='')
    print()
    print("-"*80)

    # Print rows
    for k in sorted(results.keys(), key=lambda x: (isinstance(x, str), x)):
        print(f"{str(k):<12}", end='')
        for dataset in all_datasets:
            auroc = results[k].get(dataset, 0.0)
            print(f"{auroc:<15.2f}", end='')
        print()

    print("="*80)

    # Print retention statistics
    if 1 in results and 'ALL' in results:
        print("\nPerformance Retention at k=1:")
        print("-"*80)
        for dataset in all_datasets:
            if dataset in results[1] and dataset in results['ALL']:
                auroc_1 = results[1][dataset]
                auroc_all = results['ALL'][dataset]
                retention = 100 * auroc_1 / auroc_all
                print(f"  {dataset:<20s}: {auroc_1:6.2f}% / {auroc_all:6.2f}% = {retention:5.2f}% retention")
        print("="*80)

else:
    print("No results found. Check log files manually.")
    print("="*80)

# Save to CSV
import csv
csv_file = "${OUTPUT_DIR}/topk_summary.csv"
with open(csv_file, 'w', newline='') as f:
    if results and all_datasets:
        writer = csv.writer(f)
        writer.writerow(['Top-k'] + all_datasets)
        for k in sorted(results.keys(), key=lambda x: (isinstance(x, str), x)):
            row = [str(k)]
            for dataset in all_datasets:
                row.append(results[k].get(dataset, 0.0))
            writer.writerow(row)
        print(f"\nResults saved to: {csv_file}")
EOF
