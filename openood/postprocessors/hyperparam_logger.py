"""Hyperparameter search result logger for OOD detection methods."""
import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class HyperparamLogger:
    """Logger for hyperparameter search results"""

    def __init__(self, save_dir: str, method_name: str):
        """
        Args:
            save_dir: Directory to save results
            method_name: Name of the OOD detection method (e.g., 'unlearn', 'odin')
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.method_name = method_name

        # Create timestamp for this search session
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = f"{method_name}_{self.timestamp}"

        # File paths
        self.csv_path = self.save_dir / f"{self.session_name}.csv"
        self.json_path = self.save_dir / f"{self.session_name}.json"
        self.best_path = self.save_dir / f"{self.session_name}_best.json"

        # Results storage
        self.results = []
        self.best_result = None
        self.best_metric_value = float('-inf')

        # Initialize CSV with headers
        self.csv_initialized = False

    def log_result(
        self,
        hyperparams: Dict[str, Any],
        metrics: Dict[str, float],
        dataset_name: str = "unknown",
        ood_dataset_name: str = "unknown"
    ):
        """Log a single hyperparameter configuration result

        Args:
            hyperparams: Dictionary of hyperparameter names and values
            metrics: Dictionary of metric names and values (e.g., AUROC, FPR95)
            dataset_name: Name of ID dataset
            ood_dataset_name: Name of OOD dataset
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'dataset': dataset_name,
            'ood_dataset': ood_dataset_name,
            'hyperparams': hyperparams,
            'metrics': metrics,
        }

        self.results.append(result)

        # Update best result based on AUROC (or first metric if AUROC not available)
        main_metric = metrics.get('AUROC', list(metrics.values())[0] if metrics else 0)
        if main_metric > self.best_metric_value:
            self.best_metric_value = main_metric
            self.best_result = result

        # Write to CSV
        self._write_csv_row(result)

    def _write_csv_row(self, result: Dict[str, Any]):
        """Write a single result row to CSV"""
        # Flatten the nested dict for CSV
        flat_row = {
            'timestamp': result['timestamp'],
            'dataset': result['dataset'],
            'ood_dataset': result['ood_dataset'],
        }

        # Add hyperparams
        for key, value in result['hyperparams'].items():
            flat_row[f'hp_{key}'] = value

        # Add metrics
        for key, value in result['metrics'].items():
            flat_row[f'metric_{key}'] = value

        # Initialize CSV with headers on first write
        if not self.csv_initialized:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=flat_row.keys())
                writer.writeheader()
                writer.writerow(flat_row)
            self.csv_initialized = True
        else:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=flat_row.keys())
                writer.writerow(flat_row)

    def save_summary(self):
        """Save summary of all results to JSON"""
        summary = {
            'method': self.method_name,
            'timestamp': self.timestamp,
            'num_experiments': len(self.results),
            'best_result': self.best_result,
            'all_results': self.results,
        }

        # Save full results
        with open(self.json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save best result separately
        if self.best_result:
            with open(self.best_path, 'w') as f:
                json.dump(self.best_result, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Hyperparameter Search Summary for {self.method_name}")
        print(f"{'='*80}")
        print(f"Total experiments: {len(self.results)}")
        print(f"Results saved to: {self.save_dir}")
        print(f"  - CSV: {self.csv_path.name}")
        print(f"  - JSON: {self.json_path.name}")
        print(f"  - Best: {self.best_path.name}")

        if self.best_result:
            print(f"\nBest Configuration:")
            print(f"  Dataset: {self.best_result['dataset']} vs {self.best_result['ood_dataset']}")
            print(f"  Best Metric Value: {self.best_metric_value:.4f}")
            print(f"\n  Hyperparameters:")
            for key, value in self.best_result['hyperparams'].items():
                print(f"    {key}: {value}")
            print(f"\n  Metrics:")
            for key, value in self.best_result['metrics'].items():
                print(f"    {key}: {value:.4f}")

        print(f"{'='*80}\n")

    def load_results(self, json_path: str) -> Dict[str, Any]:
        """Load previous search results from JSON

        Args:
            json_path: Path to the JSON file

        Returns:
            Dictionary containing the search results
        """
        with open(json_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def compare_results(result_paths: List[str], metric_name: str = 'AUROC'):
        """Compare results from multiple search sessions

        Args:
            result_paths: List of paths to result JSON files
            metric_name: Metric to compare (default: AUROC)
        """
        print(f"\n{'='*80}")
        print(f"Comparing Hyperparameter Search Results ({metric_name})")
        print(f"{'='*80}\n")

        comparisons = []
        for path in result_paths:
            with open(path, 'r') as f:
                data = json.load(f)

            if data.get('best_result'):
                best = data['best_result']
                metric_value = best['metrics'].get(metric_name, 0)
                comparisons.append({
                    'path': path,
                    'method': data.get('method', 'unknown'),
                    'timestamp': data.get('timestamp', 'unknown'),
                    'metric_value': metric_value,
                    'hyperparams': best['hyperparams'],
                })

        # Sort by metric value
        comparisons.sort(key=lambda x: x['metric_value'], reverse=True)

        # Print comparison
        for i, comp in enumerate(comparisons, 1):
            print(f"{i}. {comp['method']} ({comp['timestamp']})")
            print(f"   {metric_name}: {comp['metric_value']:.4f}")
            print(f"   File: {Path(comp['path']).name}")
            print(f"   Top hyperparams: {list(comp['hyperparams'].items())[:3]}")
            print()

        print(f"{'='*80}\n")


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = HyperparamLogger(save_dir="./hyperparam_results", method_name="unlearn")

    # Example: Log some results
    for i in range(5):
        hyperparams = {
            'eta': 0.001 * (i + 1),
            'num_steps': 5,
            'temp': 2.0,
            'score_type': 'delta_energy',
        }

        metrics = {
            'AUROC': 0.90 + i * 0.01,
            'FPR95': 0.15 - i * 0.01,
            'AUPR': 0.88 + i * 0.01,
        }

        logger.log_result(
            hyperparams=hyperparams,
            metrics=metrics,
            dataset_name="cifar10",
            ood_dataset_name="cifar100"
        )

    # Save summary
    logger.save_summary()
