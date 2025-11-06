"""
Example script for evaluating Circulation postprocessor on CIFAR-10.

This demonstrates gradient field circulation-based OOD detection.
The method computes circulation (line integral of gradient) along closed
curves in input space to detect OOD samples.

Usage:
    python scripts/ood/cifar10_circulation_example.py \
        --root results/cifar10_resnet18_32x32_base_e100_lr0.1 \
        --save-csv
"""

import os
import sys
ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
sys.path.append(ROOT_DIR)

import argparse
import torch

from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32

parser = argparse.ArgumentParser()
parser.add_argument('--root',
                    required=True,
                    help='Path to model checkpoint directory')
parser.add_argument('--batch-size',
                    type=int,
                    default=200,
                    help='Batch size for evaluation (smaller for circulation due to memory)')
parser.add_argument('--save-csv',
                    action='store_true',
                    help='Save results to CSV file')
parser.add_argument('--fsood',
                    action='store_true',
                    help='Use full-spectrum OOD datasets')
args = parser.parse_args()

# Configuration
id_data_name = 'cifar10'
num_classes = 10
postprocessor_name = 'circulation'

# Load model
net = ResNet18_32x32(num_classes=num_classes)
checkpoint_path = os.path.join(args.root, 's0', 'best.ckpt')

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f'Checkpoint not found at {checkpoint_path}')

checkpoint = torch.load(checkpoint_path, map_location='cpu')
net.load_state_dict(checkpoint)
net.cuda()
net.eval()

print(f'Loaded model from {checkpoint_path}')
print(f'Evaluating with {postprocessor_name} postprocessor on {id_data_name}')

# Create evaluator
evaluator = Evaluator(
    net,
    id_name=id_data_name,
    data_root=os.path.join(ROOT_DIR, 'data'),
    config_root=os.path.join(ROOT_DIR, 'configs'),
    preprocessor=None,
    postprocessor_name=postprocessor_name,
    postprocessor=None,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4
)

# Run evaluation
print('\nRunning evaluation...')
print('Note: Circulation computation is slower than standard methods due to gradient calculations.')
metrics = evaluator.eval_ood(fsood=args.fsood)

# Print results
print('\n' + '='*60)
print('EVALUATION RESULTS')
print('='*60)
print(f'Method: {postprocessor_name}')
print(f'ID Dataset: {id_data_name}')
print(f'Model: {args.root}')
print('-'*60)

# Print metrics for each OOD dataset
for dataset_name, dataset_metrics in metrics.items():
    if dataset_name == 'id_acc':
        print(f'ID Accuracy: {dataset_metrics:.2f}%')
    else:
        print(f'\nOOD Dataset: {dataset_name}')
        print(f'  AUROC: {dataset_metrics["auroc"]:.2f}%')
        print(f'  AUPR:  {dataset_metrics["aupr"]:.2f}%')
        print(f'  FPR95: {dataset_metrics["fpr95"]:.2f}%')

print('='*60)

# Save to CSV if requested
if args.save_csv:
    csv_path = os.path.join(args.root, f'{postprocessor_name}_results.csv')
    # Convert metrics to DataFrame and save
    import pandas as pd

    results = []
    for dataset_name, dataset_metrics in metrics.items():
        if dataset_name != 'id_acc':
            results.append({
                'OOD_Dataset': dataset_name,
                'AUROC': dataset_metrics['auroc'],
                'AUPR': dataset_metrics['aupr'],
                'FPR95': dataset_metrics['fpr95']
            })

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f'\nResults saved to {csv_path}')

print('\nDone!')
