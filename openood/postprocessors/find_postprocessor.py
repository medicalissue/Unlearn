from typing import Any
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


class FInDPostprocessor(BasePostprocessor):
    """Fisher Energy-based OOD detection using test-time gradients.

    Computes S(x) = g^T F^{-p} g where:
    - g is the gradient of NLL loss w.r.t. FC layer parameters
    - F is the Fisher Information Matrix (diagonal approximation)
    - p is the fisher_power parameter
    """

    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.num_classes = num_classes_dict[self.config.dataset.name]

        # Fisher Energy hyperparameters
        self.fisher_power = float(self.args.get("fisher_power", 1.0))

        # Adaptive Fisher Power hyperparameters
        self.use_adaptive_power = bool(self.args.get("use_adaptive_power", False))
        self.adaptive_metric = str(self.args.get("adaptive_metric", "entropy"))  # "entropy" or "msp"
        self.adaptive_alpha = float(self.args.get("adaptive_alpha", 1.0))

        # Aggregation method selection
        self.aggregation_method = str(self.args.get("aggregation_method", "logsumexp"))

        # Method-specific hyperparameters
        # For 'separated' method
        self.sep_threshold_percentile = float(self.args.get("sep_threshold_percentile", 50.0))
        self.sep_w_small = float(self.args.get("sep_w_small", 1.0))
        self.sep_w_large = float(self.args.get("sep_w_large", 1.0))
        self.sep_large_power = float(self.args.get("sep_large_power", 1.0))

        # For 'multiscale' method
        self.ms_percentiles = list(self.args.get("ms_percentiles", [90, 50, 10]))
        self.ms_weights = list(self.args.get("ms_weights", [1.0, 1.0, 1.0]))

        # For 'normalized' method
        self.norm_num_groups = int(self.args.get("norm_num_groups", 5))

        # For 'bayesian' method
        self.bayes_uncertainty_weight = float(self.args.get("bayes_uncertainty_weight", 1.0))
        self.bayes_confidence_weight = float(self.args.get("bayes_confidence_weight", 1.0))

        # For 'dualpath' method
        self.dual_alpha = float(self.args.get("dual_alpha", 0.5))
        self.dual_beta = float(self.args.get("dual_beta", 0.5))
        self.dual_aggregator = str(self.args.get("dual_aggregator", "median"))  # "median", "mean", "logsumexp"

        # For 'adaptive_weight' method
        self.aw_temperature = float(self.args.get("aw_temperature", 1.0))

        # Gradient type selection
        self.gradient_type = str(self.args.get("gradient_type", "nll"))

        # APS hyperparameter search space
        self.args_dict = self.config.postprocessor.postprocessor_sweep

        # Will be set in setup()
        self.fisher_W_tensor = None  # Fisher matrix for weights (global)
        self.fisher_b_tensor = None  # Fisher matrix for bias (global)
        self.model_arch = None  # Model architecture name (set in setup)

    def _get_fisher_cache_path(self):
        """Get path to cached Fisher matrix file."""
        # Get dataset name
        dataset_name = self.config.dataset.name if hasattr(self.config, 'dataset') else 'default'

        # Get model architecture name
        model_name = self.model_arch if self.model_arch else 'unknown'

        # Create filename with dataset and model
        filename = f"fisher_{dataset_name}_{model_name}.pt"

        # Try multiple sources for checkpoint path
        checkpoint_path = None
        if hasattr(self.config, 'network') and self.config.network is not None:
            checkpoint_path = getattr(self.config.network, 'checkpoint', None)
        if checkpoint_path is None and hasattr(self.config, 'ckpt_path'):
            checkpoint_path = self.config.ckpt_path

        # If no checkpoint path found, use a default cache directory
        if checkpoint_path is None:
            cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'openood', 'fisher_matrices')
            os.makedirs(cache_dir, exist_ok=True)
            return os.path.join(cache_dir, filename)

        # Use checkpoint directory if available
        checkpoint_dir = os.path.dirname(checkpoint_path)
        fisher_dir = os.path.join(checkpoint_dir, 'fisher_matrices')
        os.makedirs(fisher_dir, exist_ok=True)
        return os.path.join(fisher_dir, filename)

    def _load_fisher_matrix(self, cache_path):
        """Load Fisher matrix from cache.

        Returns:
            bool: True if successfully loaded, False otherwise
        """
        if not os.path.exists(cache_path):
            return False

        try:
            cache = torch.load(cache_path, map_location='cuda')

            # Verify compatibility
            if cache.get('num_classes') != self.num_classes:
                print(f"Warning: Cached num_classes ({cache.get('num_classes')}) doesn't match current ({self.num_classes})")
                return False
            if cache.get('dataset_name') != self.config.dataset.name:
                print(f"Warning: Cached dataset ({cache.get('dataset_name')}) doesn't match current ({self.config.dataset.name})")
                return False
            if cache.get('model_arch') != self.model_arch:
                print(f"Warning: Cached model ({cache.get('model_arch')}) doesn't match current ({self.model_arch})")
                return False

            self.fisher_W_tensor = cache['fisher_W'].cuda()
            self.fisher_b_tensor = cache['fisher_b'].cuda() if cache['fisher_b'] is not None else None

            print(f"✓ Loaded Fisher matrix from cache: {cache_path}")
            print(f"  Dataset: {cache.get('dataset_name')}, Model: {cache.get('model_arch')}")
            return True
        except Exception as e:
            print(f"Warning: Failed to load Fisher cache: {e}")
            return False

    def _save_fisher_matrix(self, cache_path):
        """Save Fisher matrix to cache."""
        try:
            cache = {
                'fisher_W': self.fisher_W_tensor.cpu(),
                'fisher_b': self.fisher_b_tensor.cpu() if self.fisher_b_tensor is not None else None,
                'num_classes': self.num_classes,
                'dataset_name': self.config.dataset.name,
                'model_arch': self.model_arch,
                'feature_dim': self.fisher_W_tensor.shape[-1]
            }
            torch.save(cache, cache_path)
            print(f"✓ Saved Fisher matrix to cache: {cache_path}")
            print(f"  Dataset: {self.config.dataset.name}, Model: {self.model_arch}")
        except Exception as e:
            print(f"Warning: Failed to save Fisher cache: {e}")

    def _compute_gradient_logits(self, logits, labels=None):
        """Compute gradient of various metrics w.r.t. logits.

        Args:
            logits: [batch, num_classes] tensor
            labels: [batch] tensor of labels (required for NLL)

        Returns:
            grad_logits: [batch, num_classes] gradient tensor
        """
        probs = F.softmax(logits, dim=-1)  # [batch, num_classes]
        eps = 1e-10

        if self.gradient_type == "nll":
            # Gradient of NLL: ∇_z L = p - e_y
            if labels is None:
                raise ValueError("Labels required for NLL gradient")
            grad_logits = probs.clone()
            grad_logits[torch.arange(len(labels)), labels] -= 1.0

        elif self.gradient_type == "entropy":
            # Gradient of entropy: ∇_z H(p) = p * (log(p) + 1) - p
            # Simplified: ∇_z H = p * log(p)
            log_probs = torch.log(probs + eps)
            grad_logits = probs * log_probs  # [batch, num_classes]

        elif self.gradient_type == "kl":
            # Gradient of KL divergence from target: ∇_z KL(p||target)
            # where target is one-hot at predicted/true class
            # ∇KL = p - target (same form as NLL)
            if labels is None:
                raise ValueError("Labels required for KL gradient")
            target_dist = torch.zeros_like(probs)
            target_dist[torch.arange(len(labels)), labels] = 1.0
            grad_logits = probs - target_dist  # [batch, num_classes]

        elif self.gradient_type == "energy":
            # Gradient of energy-weighted NLL: ∇_z (E(x) * NLL)
            # E(x) = log(sum(exp(z))) - energy score
            # Weighted by energy to emphasize confident predictions
            if labels is None:
                raise ValueError("Labels required for energy gradient")
            energy = torch.logsumexp(logits, dim=-1, keepdim=True)  # [batch, 1]
            grad_nll = probs.clone()
            grad_nll[torch.arange(len(labels)), labels] -= 1.0
            grad_logits = energy * grad_nll  # [batch, num_classes]

        elif self.gradient_type == "hellinger":
            # Gradient of Hellinger distance from target: ∇_z H(p, target)
            # H² = 1/2 * sum((sqrt(p_i) - sqrt(target_i))²)
            # Use predicted label as target for supervised signal
            if labels is None:
                raise ValueError("Labels required for hellinger gradient")
            target = torch.zeros_like(probs)
            target[torch.arange(len(labels)), labels] = 1.0
            sqrt_probs = torch.sqrt(probs + eps)
            sqrt_target = torch.sqrt(target + eps)

            # Simplified Hellinger gradient
            diff = sqrt_probs - sqrt_target  # [batch, num_classes]
            grad_logits = diff * probs  # [batch, num_classes]

        else:
            raise ValueError(f"Unknown gradient_type: {self.gradient_type}")

        return grad_logits

    def _get_fc_layer(self, net: nn.Module):
        """Get the final classification layer from the network.

        Supports various naming conventions: fc, head, heads, classifier.
        """
        # Try common attribute names
        for attr_name in ['fc', 'head', 'heads', 'classifier']:
            if hasattr(net, attr_name):
                layer = getattr(net, attr_name)
                # Ensure it's a Linear layer
                if isinstance(layer, nn.Linear):
                    return layer
                # For Sequential/ModuleList, get the last Linear layer
                elif isinstance(layer, (nn.Sequential, nn.ModuleList)):
                    for module in reversed(list(layer.modules())):
                        if isinstance(module, nn.Linear):
                            return module

        # Try net.module for DataParallel
        if hasattr(net, 'module'):
            return self._get_fc_layer(net.module)

        # Last resort: search all modules for the last Linear layer
        linear_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
        if linear_layers:
            return linear_layers[-1]

        raise ValueError("Cannot find FC layer in the network")

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """Compute or load Fisher Information Matrix from ID training data."""
        import tqdm
        net.eval()

        # Get model architecture name
        self.model_arch = net.__class__.__name__

        # Try to load from cache first
        cache_path = self._get_fisher_cache_path()
        if self._load_fisher_matrix(cache_path):
            return  # Successfully loaded from cache

        # Cache miss - compute Fisher matrix
        print(f"Fisher matrix cache not found, computing from scratch...")

        # Get ID training loader
        if 'train' in id_loader_dict:
            id_loader = id_loader_dict['train']
        else:
            print("Warning: No 'train' loader found, using 'val' loader")
            id_loader = id_loader_dict['val']

        # Get FC layer
        fc = self._get_fc_layer(net)

        fc_weight = fc.weight.data.clone()
        fc_bias = fc.bias.data.clone() if fc.bias is not None else None
        has_bias = fc_bias is not None

        # Initialize Fisher accumulators (global mode)
        fisher_W_acc = torch.zeros_like(fc_weight)
        if has_bias:
            fisher_b_acc = torch.zeros_like(fc_bias)
        total_count = 0

        # Compute Fisher matrix
        print(f"Computing Fisher Information Matrix...")
        for batch in tqdm.tqdm(id_loader, desc="Fisher Matrix"):
            data = batch['data'].cuda()
            labels = batch['label'].cuda()

            with torch.no_grad():
                # Extract features and logits
                output = net(data, return_feature=True)
                if isinstance(output, tuple):
                    _, features = output  # (logits, features)
                else:
                    features = output

                # Compute logits
                logits = F.linear(features, fc_weight, fc_bias if has_bias else None)

                # Compute gradient w.r.t. logits using selected gradient type
                grad_logits = self._compute_gradient_logits(logits, labels)  # [batch_size, num_classes]

                # Compute gradient w.r.t. FC weight: ∇_W L = grad_logits^T @ features
                # Shape: [batch_size, num_classes, feature_dim]
                grad_W_batch = torch.einsum('bc,bd->bcd', grad_logits, features)

                # Compute gradient w.r.t. FC bias: ∇_b L = grad_logits
                if has_bias:
                    grad_b_batch = grad_logits  # [batch_size, num_classes]
                else:
                    grad_b_batch = None

            # Accumulate squared gradients (Fisher approximation: E[g g^T] ≈ g^2)
            fisher_W_acc += torch.sum(grad_W_batch ** 2, dim=0)
            if has_bias:
                fisher_b_acc += torch.sum(grad_b_batch ** 2, dim=0)
            total_count += len(labels)

        # Average Fisher matrices
        fisher_W_acc /= total_count
        self.fisher_W_tensor = fisher_W_acc
        if has_bias:
            fisher_b_acc /= total_count
            self.fisher_b_tensor = fisher_b_acc

        print(f"Fisher Information Matrix computed")

        # Save to cache
        self._save_fisher_matrix(cache_path)

    def _aggregate_logsumexp(self, log_terms):
        """LogSumExp aggregation (original method).

        Args:
            log_terms: [batch, dim] tensor of log(g²/F^p)

        Returns:
            scores: [batch] tensor
        """
        max_log = torch.max(log_terms, dim=1, keepdim=True)[0]
        log_energy = max_log.squeeze(1) + torch.log(
            torch.sum(torch.exp(log_terms - max_log), dim=1)
        )
        return log_energy

    def _aggregate_separated(self, log_terms, F_vec, fisher_power_adaptive):
        """Separated scoring: split small-F and large-F dimensions.

        Args:
            log_terms: [batch, dim] tensor of log(g²/F^p)
            F_vec: [dim] Fisher matrix vector
            fisher_power_adaptive: [batch] or scalar

        Returns:
            scores: [batch] tensor
        """
        # Determine threshold
        F_threshold = torch.quantile(F_vec, self.sep_threshold_percentile / 100.0)

        # Create masks
        small_F_mask = F_vec < F_threshold  # [dim]
        large_F_mask = ~small_F_mask  # [dim]

        # Small F dimensions: use inverse Fisher (g²/F^p)
        if small_F_mask.any():
            score_small = torch.mean(log_terms[:, small_F_mask], dim=1)  # [batch]
        else:
            score_small = torch.zeros(log_terms.shape[0], device=log_terms.device)

        # Large F dimensions: use forward Fisher (g²*F^q)
        if large_F_mask.any():
            # Recompute with positive power for large F
            g_squared_large = torch.exp(log_terms[:, large_F_mask]) * (F_vec[large_F_mask] ** fisher_power_adaptive.unsqueeze(1))
            log_g_squared_F = torch.log(g_squared_large + 1e-10)  # log(g²*F^p)
            score_large = torch.mean(log_g_squared_F, dim=1)  # [batch]
        else:
            score_large = torch.zeros(log_terms.shape[0], device=log_terms.device)

        # Weighted combination (note: minus sign for forward Fisher to match convention)
        total_score = self.sep_w_small * score_small - self.sep_w_large * score_large
        return total_score

    def _aggregate_multiscale(self, log_terms):
        """Multi-scale percentile aggregation.

        Args:
            log_terms: [batch, dim] tensor of log(g²/F^p)

        Returns:
            scores: [batch] tensor
        """
        scores = []
        for percentile, weight in zip(self.ms_percentiles, self.ms_weights):
            p_score = torch.quantile(log_terms, percentile / 100.0, dim=1)  # [batch]
            scores.append(weight * p_score)

        total_score = torch.stack(scores, dim=0).sum(dim=0)  # [batch]
        return total_score

    def _aggregate_normalized(self, log_terms, F_vec):
        """Normalized weighted sum by Fisher value groups.

        Args:
            log_terms: [batch, dim] tensor of log(g²/F^p)
            F_vec: [dim] Fisher matrix vector

        Returns:
            scores: [batch] tensor
        """
        # Sort dimensions by Fisher value
        sorted_indices = torch.argsort(F_vec)
        group_size = len(F_vec) // self.norm_num_groups

        group_scores = []
        for i in range(self.norm_num_groups):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < self.norm_num_groups - 1 else len(F_vec)
            group_indices = sorted_indices[start_idx:end_idx]

            # Normalize within group
            group_terms = log_terms[:, group_indices]  # [batch, group_size]
            group_mean = torch.mean(group_terms, dim=1, keepdim=True)
            group_std = torch.std(group_terms, dim=1, keepdim=True) + 1e-6
            normalized = (group_terms - group_mean) / group_std

            # Aggregate normalized terms
            group_score = torch.mean(normalized, dim=1)  # [batch]
            group_scores.append(group_score)

        # Average across groups
        total_score = torch.stack(group_scores, dim=0).mean(dim=0)  # [batch]
        return total_score

    def _aggregate_bayesian(self, log_terms, F_vec, g_batch):
        """Bayesian information combination.

        Args:
            log_terms: [batch, dim] tensor of log(g²/F^p)
            F_vec: [dim] Fisher matrix vector
            g_batch: [batch, dim] gradient vector

        Returns:
            scores: [batch] tensor
        """
        # Uncertainty score: dominated by small F (high uncertainty)
        # Use logsumexp of inverse terms (higher = more OOD in flat directions)
        uncertainty_score = self._aggregate_logsumexp(log_terms)  # [batch]

        # Confidence score: weighted by large F (high confidence)
        # Compute Fisher-weighted gradient magnitude (higher = more OOD in curved directions)
        g_squared = g_batch ** 2  # [batch, dim]
        F_weighted = g_squared * F_vec.unsqueeze(0)  # [batch, dim]
        confidence_score = torch.log(torch.sum(F_weighted, dim=1) + 1e-10)  # [batch]

        # Combine (both positive: higher = more OOD)
        total_score = (self.bayes_uncertainty_weight * uncertainty_score +
                      self.bayes_confidence_weight * confidence_score)
        return total_score

    def _aggregate_dualpath(self, log_terms, F_vec, g_batch, fisher_power_adaptive):
        """Dual-path: separate inverse and forward Fisher paths.

        Args:
            log_terms: [batch, dim] tensor of log(g²/F^p)
            F_vec: [dim] Fisher matrix vector
            g_batch: [batch, dim] gradient vector
            fisher_power_adaptive: [batch] or scalar

        Returns:
            scores: [batch] tensor
        """
        # Path 1: Inverse Fisher (g²/F^p) - captures flat direction anomalies
        if self.dual_aggregator == "logsumexp":
            score_inv = self._aggregate_logsumexp(log_terms)
        elif self.dual_aggregator == "median":
            score_inv = torch.median(log_terms, dim=1)[0]
        elif self.dual_aggregator == "mean":
            score_inv = torch.mean(log_terms, dim=1)
        else:
            raise ValueError(f"Unknown dual_aggregator: {self.dual_aggregator}")

        # Path 2: Forward Fisher (g²*F^p) - captures curvature direction anomalies
        g_squared = g_batch ** 2  # [batch, dim]
        F_powered = F_vec ** fisher_power_adaptive.unsqueeze(1)  # [batch, dim] or broadcast
        forward_terms = g_squared * F_powered  # [batch, dim]
        log_forward = torch.log(forward_terms + 1e-10)

        if self.dual_aggregator == "logsumexp":
            score_fwd = self._aggregate_logsumexp(log_forward)
        elif self.dual_aggregator == "median":
            score_fwd = torch.median(log_forward, dim=1)[0]
        elif self.dual_aggregator == "mean":
            score_fwd = torch.mean(log_forward, dim=1)
        else:
            raise ValueError(f"Unknown dual_aggregator: {self.dual_aggregator}")

        # Combine paths (note: minus sign for forward path to match convention)
        total_score = self.dual_alpha * score_inv - self.dual_beta * score_fwd
        return total_score

    def _aggregate_adaptive_weight(self, log_terms, F_vec):
        """Adaptive Fisher weighting with sigmoid function.

        Args:
            log_terms: [batch, dim] tensor of log(g²/F^p)
            F_vec: [dim] Fisher matrix vector

        Returns:
            scores: [batch] tensor
        """
        # Normalize Fisher values to [-1, 1] range
        log_F = torch.log(F_vec + 1e-8)
        F_mean = torch.mean(log_F)
        F_std = torch.std(log_F) + 1e-6
        F_normalized = (log_F - F_mean) / F_std  # [dim]

        # Compute adaptive weights using sigmoid
        # Small F (negative normalized) → weight ≈ 0.5-1.0
        # Large F (positive normalized) → weight ≈ 0.0-0.5
        weights = torch.sigmoid(-F_normalized / self.aw_temperature)  # [dim]

        # Weight the log terms
        weighted_terms = weights.unsqueeze(0) * log_terms  # [batch, dim]

        # Aggregate
        total_score = torch.sum(weighted_terms, dim=1)  # [batch]
        return total_score

    def _compute_fisher_score(self, g_batch, F_vec, fisher_power_adaptive):
        """Compute Fisher-based OOD score using selected aggregation method.

        Args:
            g_batch: [batch, dim] gradient tensor
            F_vec: [dim] Fisher matrix vector
            fisher_power_adaptive: [batch] adaptive power values

        Returns:
            scores: [batch] OOD scores (higher = more OOD)
        """
        # Compute log terms: log(g²/F^p) = 2*log|g| - p*log(F)
        eps_g = 1e-10
        eps_f = 1e-8
        log_g_squared = 2.0 * torch.log(torch.abs(g_batch) + eps_g)  # [batch, dim]
        log_F_powered = fisher_power_adaptive.unsqueeze(1) * torch.log(F_vec + eps_f).unsqueeze(0)  # [batch, dim]
        log_terms = log_g_squared - log_F_powered  # [batch, dim]

        # Select aggregation method
        if self.aggregation_method == "logsumexp":
            score = self._aggregate_logsumexp(log_terms)
        elif self.aggregation_method == "separated":
            score = self._aggregate_separated(log_terms, F_vec, fisher_power_adaptive)
        elif self.aggregation_method == "multiscale":
            score = self._aggregate_multiscale(log_terms)
        elif self.aggregation_method == "normalized":
            score = self._aggregate_normalized(log_terms, F_vec)
        elif self.aggregation_method == "bayesian":
            score = self._aggregate_bayesian(log_terms, F_vec, g_batch)
        elif self.aggregation_method == "dualpath":
            score = self._aggregate_dualpath(log_terms, F_vec, g_batch, fisher_power_adaptive)
        elif self.aggregation_method == "adaptive_weight":
            score = self._aggregate_adaptive_weight(log_terms, F_vec)
        else:
            raise ValueError(f"Unknown aggregation_method: {self.aggregation_method}")

        return score


    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """Compute OOD scores for a batch of test samples.

        Args:
            net: Neural network
            data: Batch of test samples (tensor)

        Returns:
            pred: Class predictions [batch_size]
            conf: OOD confidence scores [batch_size] (higher = more OOD)
        """
        # Get FC layer
        fc = self._get_fc_layer(net)

        # Extract features and logits
        output = net(data, return_feature=True)
        if isinstance(output, tuple):
            logits, features = output  # (logits, features)
        else:
            features = output
            logits = fc(features)

        # Predictions
        pred = logits.argmax(dim=1).cpu()
        pred_labels = pred.cuda()  # Keep on GPU for gradient computation

        # Compute gradient w.r.t. logits using selected gradient type
        grad_logits = self._compute_gradient_logits(logits, pred_labels)  # [batch, num_classes]

        # Compute gradient w.r.t. FC parameters (batched)
        # ∇_W L = grad_logits^T ⊗ features
        grad_W_batch = torch.einsum('bc,bd->bcd', grad_logits, features)  # [batch, num_classes, feature_dim]

        has_bias = fc.bias is not None
        if has_bias:
            grad_b_batch = grad_logits  # [batch, num_classes]
            # Flatten and concatenate
            g_batch = torch.cat([
                grad_W_batch.reshape(len(features), -1),  # [batch, num_classes * feature_dim]
                grad_b_batch  # [batch, num_classes]
            ], dim=1)  # [batch, num_classes * feature_dim + num_classes]

            # Fisher matrix (flattened)
            F_vec = torch.cat([
                self.fisher_W_tensor.flatten(),
                self.fisher_b_tensor.flatten()
            ])  # [num_classes * feature_dim + num_classes]
        else:
            # No bias
            g_batch = grad_W_batch.reshape(len(features), -1)  # [batch, num_classes * feature_dim]
            F_vec = self.fisher_W_tensor.flatten()  # [num_classes * feature_dim]

        # Adaptive Fisher Power: adjust power based on prediction uncertainty metric
        if self.use_adaptive_power:
            probs = F.softmax(logits, dim=-1)  # [batch, num_classes]

            if self.adaptive_metric == "entropy":
                # Compute entropy: H(x) = -sum(p * log(p))
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [batch]
                max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32))
                metric_norm = entropy / max_entropy  # Normalized entropy [0, 1]
            elif self.adaptive_metric == "msp":
                # Use MSP (Maximum Softmax Probability): 1 - max(p)
                # High MSP (low uncertainty) → Low metric value
                # Low MSP (high uncertainty) → High metric value
                max_prob = torch.max(probs, dim=-1)[0]  # [batch]
                metric_norm = 1.0 - max_prob  # [batch], range [0, 1]
            else:
                raise ValueError(f"Unknown adaptive_metric: {self.adaptive_metric}")

            # Adaptive power: p*(x) = p_base * (1 + α*M)
            # High uncertainty → High power (favors far-OOD detection)
            fisher_power_adaptive = self.fisher_power * (1.0 + self.adaptive_alpha * metric_norm)
        else:
            # Use fixed fisher power for all samples
            fisher_power_adaptive = torch.full((len(features),), self.fisher_power, device=g_batch.device)  # [batch]

        # Compute Fisher-based OOD score using selected aggregation method
        log_energy = self._compute_fisher_score(g_batch, F_vec, fisher_power_adaptive)

        # Negated log-energy (higher = more OOD)
        conf = -log_energy.cpu()

        return pred, conf

    def set_hyperparam(self, hyperparam: list):
        """Set hyperparameters for APS mode.

        Args:
            hyperparam: List containing [fisher_power] or [fisher_power, adaptive_alpha]
        """
        self.fisher_power = hyperparam[0]
        if len(hyperparam) > 1:
            self.adaptive_alpha = hyperparam[1]

    def get_hyperparam(self):
        """Get current hyperparameters for APS mode.

        Returns:
            List containing [fisher_power] or [fisher_power, adaptive_alpha] if adaptive mode enabled
        """
        if self.use_adaptive_power:
            return [self.fisher_power, self.adaptive_alpha]
        else:
            return [self.fisher_power]
