from typing import Any
import os
import numpy as np
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
        self.adaptive_alpha = float(self.args.get("adaptive_alpha", 1.0))

        # Top-k Fisher Selection hyperparameters
        self.use_topk = bool(self.args.get("use_topk", False))
        self.topk = int(self.args.get("topk", 1000))

        # Gradient type configurations
        self.fisher_gradient_type = str(self.args.get("fisher_gradient_type", "nll"))
        self.test_gradient_type = str(self.args.get("test_gradient_type", "nll"))

        # Focal loss hyperparameters
        self.focal_gamma = float(self.args.get("focal_gamma", 2.0))
        self.focal_alpha = float(self.args.get("focal_alpha", 1.0))

        # Label smoothing hyperparameters
        self.smoothing = float(self.args.get("smoothing", 0.1))

        # Validate gradient types
        valid_types = ["nll", "entropy", "focal", "label_smoothing", "kl"]
        if self.fisher_gradient_type not in valid_types:
            raise ValueError(f"Invalid fisher_gradient_type: {self.fisher_gradient_type}. Must be one of {valid_types}")
        if self.test_gradient_type not in valid_types:
            raise ValueError(f"Invalid test_gradient_type: {self.test_gradient_type}. Must be one of {valid_types}")

        # APS hyperparameter search space
        self.args_dict = self.config.postprocessor.postprocessor_sweep

        # Will be set in setup()
        self.fisher_W_tensor = None  # Fisher matrix for weights (global)
        self.fisher_b_tensor = None  # Fisher matrix for bias (global)
        self.model_arch = None  # Model architecture name (set in setup)
        self.device = None  # Device will be set from model in setup()

        # Gradient statistics collection (for analysis)
        self.collect_gradient_stats = False
        self.gradient_stats = []

    def _get_fisher_cache_path(self):
        """Get path to cached Fisher matrix file.

        Filename includes: dataset, model, gradient_type, and relevant hyperparameters.
        """
        # Get dataset name
        dataset_name = self.config.dataset.name if hasattr(self.config, 'dataset') else 'default'

        # Get model architecture name
        model_name = self.model_arch if self.model_arch else 'unknown'

        # Try multiple sources for checkpoint path
        checkpoint_path = None
        if hasattr(self.config, 'network') and self.config.network is not None:
            checkpoint_path = getattr(self.config.network, 'checkpoint', None)
        if checkpoint_path is None and hasattr(self.config, 'ckpt_path'):
            checkpoint_path = self.config.ckpt_path

        # Extract checkpoint filename (without extension) if available
        ckpt_name = ""
        if checkpoint_path:
            ckpt_filename = os.path.basename(checkpoint_path)
            # Remove extension (.pth, .pt, .ckpt, etc.)
            ckpt_name = os.path.splitext(ckpt_filename)[0]

        # Build hyperparameter string based on gradient type
        grad_type = self.fisher_gradient_type
        if grad_type == "focal":
            hyperparam_str = f"gamma{self.focal_gamma}_alpha{self.focal_alpha}"
        elif grad_type == "label_smoothing":
            hyperparam_str = f"smooth{self.smoothing}"
        else:
            # For nll, entropy, kl: no additional hyperparameters
            hyperparam_str = ""

        # Create filename: fisher_{dataset}_{model}_{ckpt_name}_{grad_type}[_{hyperparams}].pt
        if ckpt_name:
            if hyperparam_str:
                filename = f"fisher_{dataset_name}_{model_name}_{ckpt_name}_{grad_type}_{hyperparam_str}.pt"
            else:
                filename = f"fisher_{dataset_name}_{model_name}_{ckpt_name}_{grad_type}.pt"
        else:
            # No checkpoint path, use simpler naming
            if hyperparam_str:
                filename = f"fisher_{dataset_name}_{model_name}_{grad_type}_{hyperparam_str}.pt"
            else:
                filename = f"fisher_{dataset_name}_{model_name}_{grad_type}.pt"

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
            # Load to CPU first, then move to the correct device
            cache = torch.load(cache_path, map_location='cpu')

            # Verify compatibility: basic parameters
            if cache.get('num_classes') != self.num_classes:
                print(f"Warning: Cached num_classes ({cache.get('num_classes')}) doesn't match current ({self.num_classes})")
                return False
            if cache.get('dataset_name') != self.config.dataset.name:
                print(f"Warning: Cached dataset ({cache.get('dataset_name')}) doesn't match current ({self.config.dataset.name})")
                return False
            if cache.get('model_arch') != self.model_arch:
                print(f"Warning: Cached model ({cache.get('model_arch')}) doesn't match current ({self.model_arch})")
                return False

            # Verify compatibility: gradient configuration
            grad_config = cache.get('gradient_config', {})
            cached_grad_type = grad_config.get('fisher_gradient_type', 'nll')  # Default to 'nll' for old caches

            if cached_grad_type != self.fisher_gradient_type:
                print(f"Warning: Cached gradient type ({cached_grad_type}) doesn't match current ({self.fisher_gradient_type})")
                return False

            # Verify hyperparameters for focal loss
            if self.fisher_gradient_type == "focal":
                cached_gamma = grad_config.get('focal_gamma')
                cached_alpha = grad_config.get('focal_alpha')
                if cached_gamma != self.focal_gamma or cached_alpha != self.focal_alpha:
                    print(f"Warning: Cached focal hyperparameters (gamma={cached_gamma}, alpha={cached_alpha}) "
                          f"don't match current (gamma={self.focal_gamma}, alpha={self.focal_alpha})")
                    return False

            # Verify hyperparameters for label smoothing
            if self.fisher_gradient_type == "label_smoothing":
                cached_smoothing = grad_config.get('smoothing')
                if cached_smoothing != self.smoothing:
                    print(f"Warning: Cached smoothing ({cached_smoothing}) doesn't match current ({self.smoothing})")
                    return False

            # Move to the device (will be moved to correct device in setup when self.device is set)
            self.fisher_W_tensor = cache['fisher_W']
            self.fisher_b_tensor = cache['fisher_b'] if cache['fisher_b'] is not None else None

            print(f"✓ Loaded Fisher matrix from cache: {cache_path}")
            print(f"  Dataset: {cache.get('dataset_name')}, Model: {cache.get('model_arch')}")
            print(f"  Gradient type: {cached_grad_type}")
            return True
        except Exception as e:
            print(f"Warning: Failed to load Fisher cache: {e}")
            return False

    def _save_fisher_matrix(self, cache_path):
        """Save Fisher matrix to cache."""
        try:
            # Build gradient config dictionary
            grad_config = {
                'fisher_gradient_type': self.fisher_gradient_type,
            }

            # Add hyperparameters based on gradient type
            if self.fisher_gradient_type == "focal":
                grad_config['focal_gamma'] = self.focal_gamma
                grad_config['focal_alpha'] = self.focal_alpha
            elif self.fisher_gradient_type == "label_smoothing":
                grad_config['smoothing'] = self.smoothing

            cache = {
                'fisher_W': self.fisher_W_tensor.cpu(),
                'fisher_b': self.fisher_b_tensor.cpu() if self.fisher_b_tensor is not None else None,
                'num_classes': self.num_classes,
                'dataset_name': self.config.dataset.name,
                'model_arch': self.model_arch,
                'feature_dim': self.fisher_W_tensor.shape[-1],
                'gradient_config': grad_config
            }
            torch.save(cache, cache_path)
            print(f"✓ Saved Fisher matrix to cache: {cache_path}")
            print(f"  Dataset: {self.config.dataset.name}, Model: {self.model_arch}")
            print(f"  Gradient type: {self.fisher_gradient_type}")
        except Exception as e:
            print(f"Warning: Failed to save Fisher cache: {e}")


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

        # Search all modules for the last Linear layer
        linear_layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
        if linear_layers:
            return linear_layers[-1]

        raise ValueError("Cannot find FC layer in the network")

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """Compute or load Fisher Information Matrix from ID training data."""
        import tqdm
        net.eval()

        # Detect device from model parameters
        self.device = next(net.parameters()).device
        print(f"Device: {self.device}")

        # Get model architecture name
        self.model_arch = net.__class__.__name__

        # Try to load from cache first
        cache_path = self._get_fisher_cache_path()
        print(f"Looking for Fisher matrix at: {cache_path}")
        if self._load_fisher_matrix(cache_path):
            # Move cached Fisher matrices to the correct device
            self.fisher_W_tensor = self.fisher_W_tensor.to(self.device)
            if self.fisher_b_tensor is not None:
                self.fisher_b_tensor = self.fisher_b_tensor.to(self.device)
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
            data = batch['data'].to(self.device)
            labels = batch['label'].to(self.device)

            with torch.no_grad():
                # Extract features and logits
                output = net(data, return_feature=True)
                if isinstance(output, tuple):
                    _, features = output  # (logits, features)
                else:
                    features = output

                # Compute logits and probabilities
                logits = F.linear(features, fc_weight, fc_bias if has_bias else None)
                probs = F.softmax(logits, dim=-1)

                # Compute gradient w.r.t. logits based on gradient type
                if self.fisher_gradient_type == "nll":
                    # NLL-based gradient: ∇_z L = p - e_y
                    # where p is the probability vector and e_y is one-hot label
                    grad_logits = probs.clone()  # [batch_size, num_classes]
                    grad_logits[torch.arange(len(labels)), labels] -= 1.0
                elif self.fisher_gradient_type == "entropy":
                    # Entropy-based gradient: ∇_z H = p * (log(p) + 1)
                    # where H = -sum(p * log(p)) is the entropy
                    grad_logits = probs * (torch.log(probs + 1e-10) + 1.0)  # [batch_size, num_classes]
                elif self.fisher_gradient_type == "focal":
                    # Focal loss gradient: ∇_z L_focal = α * [(1-p_t)^γ * p - γ * p_t * (1-p_t)^(γ-1) * p * (e_y - p)]
                    # Simplified: α * (1-p_t)^(γ-1) * [p * (1-p_t) - γ * p_t * (e_y - p)]
                    p_t = probs[torch.arange(len(labels)), labels]  # [batch_size]
                    one_minus_pt = 1.0 - p_t  # [batch_size]

                    # Create one-hot encoding
                    one_hot = torch.zeros_like(probs)
                    one_hot[torch.arange(len(labels)), labels] = 1.0

                    # Focal weight: (1-p_t)^(γ-1) with epsilon for numerical stability
                    focal_weight = torch.pow(one_minus_pt + 1e-10, self.focal_gamma - 1.0)  # [batch_size]

                    # Gradient: α * (1-p_t)^(γ-1) * [p * (1-p_t) - γ * p_t * (e_y - p)]
                    term1 = probs * one_minus_pt.unsqueeze(1)  # [batch_size, num_classes]
                    term2 = self.focal_gamma * p_t.unsqueeze(1) * (one_hot - probs)  # [batch_size, num_classes]
                    grad_logits = self.focal_alpha * focal_weight.unsqueeze(1) * (term1 - term2)
                elif self.fisher_gradient_type == "label_smoothing":
                    # Label smoothing gradient: ∇_z L_ls = p - [(1-ε)*e_y + ε/K]
                    # where ε is smoothing factor, K is num_classes
                    smooth_value = self.smoothing / self.num_classes

                    # Smoothed labels: (1-ε)*e_y + ε/K for all classes
                    grad_logits = probs.clone()  # [batch_size, num_classes]
                    grad_logits -= smooth_value  # Subtract ε/K from all
                    grad_logits[torch.arange(len(labels)), labels] -= (1.0 - self.smoothing)  # Additional (1-ε) for true class
                elif self.fisher_gradient_type == "kl":
                    # KL divergence from uniform distribution gradient
                    # KL(p || u) = sum(p * log(p/u)) = sum(p * log(p)) - sum(p * log(u))
                    # where u = 1/K (uniform distribution)
                    # ∇_z KL = p * (log(p) + 1 + log(K))
                    # Peaked (ID) produces large gradient → large Fisher Energy → low OOD score ✓
                    # Uniform (OOD) produces small gradient → small Fisher Energy → high OOD score ✓
                    log_K = torch.log(torch.tensor(self.num_classes, dtype=torch.float32, device=probs.device))
                    grad_logits = probs * (torch.log(probs + 1e-10) + 1.0 + log_K)  # [batch_size, num_classes]

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
        # Ensure device is set (in case postprocess is called without setup)
        if self.device is None:
            self.device = next(net.parameters()).device

        # Ensure Fisher matrices are on the correct device
        if self.fisher_W_tensor.device != self.device:
            self.fisher_W_tensor = self.fisher_W_tensor.to(self.device)
            if self.fisher_b_tensor is not None:
                self.fisher_b_tensor = self.fisher_b_tensor.to(self.device)

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

        # Compute gradient w.r.t. logits (batched) based on gradient type
        probs = F.softmax(logits, dim=-1)  # [batch, num_classes]

        if self.test_gradient_type == "nll":
            # NLL-based gradient: ∇_z L = p - e_y_pred
            # where p is the probability vector and e_y_pred is one-hot predicted label
            pred_labels = logits.argmax(dim=-1)  # [batch]
            grad_logits = probs.clone()
            grad_logits[torch.arange(len(pred_labels)), pred_labels] -= 1.0
        elif self.test_gradient_type == "entropy":
            # Entropy-based gradient: ∇_z H = p * (log(p) + 1)
            # where H = -sum(p * log(p)) is the entropy
            grad_logits = probs * (torch.log(probs + 1e-10) + 1.0)  # [batch, num_classes]
        elif self.test_gradient_type == "focal":
            # Focal loss gradient using predicted labels
            pred_labels = logits.argmax(dim=-1)  # [batch]
            p_t = probs[torch.arange(len(pred_labels)), pred_labels]  # [batch]
            one_minus_pt = 1.0 - p_t  # [batch]

            # Create one-hot encoding for predicted labels
            one_hot = torch.zeros_like(probs)
            one_hot[torch.arange(len(pred_labels)), pred_labels] = 1.0

            # Focal weight: (1-p_t)^(γ-1) with epsilon for numerical stability
            focal_weight = torch.pow(one_minus_pt + 1e-10, self.focal_gamma - 1.0)  # [batch]

            # Gradient: α * (1-p_t)^(γ-1) * [p * (1-p_t) - γ * p_t * (e_y - p)]
            term1 = probs * one_minus_pt.unsqueeze(1)  # [batch, num_classes]
            term2 = self.focal_gamma * p_t.unsqueeze(1) * (one_hot - probs)  # [batch, num_classes]
            grad_logits = self.focal_alpha * focal_weight.unsqueeze(1) * (term1 - term2)
        elif self.test_gradient_type == "label_smoothing":
            # Label smoothing gradient using predicted labels
            pred_labels = logits.argmax(dim=-1)  # [batch]
            smooth_value = self.smoothing / self.num_classes

            # Smoothed labels: (1-ε)*e_y + ε/K for all classes
            grad_logits = probs.clone()  # [batch, num_classes]
            grad_logits -= smooth_value  # Subtract ε/K from all
            grad_logits[torch.arange(len(pred_labels)), pred_labels] -= (1.0 - self.smoothing)  # Additional (1-ε) for predicted class
        elif self.test_gradient_type == "kl":
            # KL divergence from uniform distribution gradient (NOT negated at test time)
            # KL(p || u) = sum(p * log(p/u)) where u = 1/K
            # ∇_z KL = p * (log(p) + 1 + log(K))
            # Peaked (ID) gets larger magnitude gradient
            log_K = torch.log(torch.tensor(self.num_classes, dtype=torch.float32, device=probs.device))
            grad_logits = -probs * (torch.log(probs + 1e-10) + 1.0 + log_K)  # [batch, num_classes]

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

        # Adaptive Fisher Power: adjust power based on prediction entropy
        if self.use_adaptive_power:
            # Compute entropy: H(x) = -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [batch]
            max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float32))
            entropy_norm = entropy / max_entropy  # Normalized entropy [0, 1]

            # Adaptive power: p*(x) = p_base * (1 + α*H)
            # High entropy → High power (favors far-OOD detection)
            fisher_power_adaptive = self.fisher_power * (1.0 + self.adaptive_alpha * entropy_norm)
        else:
            # Use fixed fisher power for all samples
            fisher_power_adaptive = torch.full((len(features),), self.fisher_power, device=g_batch.device)  # [batch]

        # Compute Fisher Energy: S(x) = sum(g^2 / F^p) for each sample
        # Log-space: log(g^2 / F^p) = 2*log|g| - p*log(F)
        eps_g = 1e-10
        eps_f = 1e-8
        log_g_squared = 2.0 * torch.log(torch.abs(g_batch) + eps_g)  # [batch, dim]
        log_F_powered = fisher_power_adaptive.unsqueeze(1) * torch.log(F_vec + eps_f).unsqueeze(0)  # [batch, dim]
        log_terms = log_g_squared - log_F_powered  # [batch, dim]

        # Apply Top-k Fisher selection if enabled
        if self.use_topk:
            # Select Top-k smallest Fisher values (dimension with largest 1/F^p contribution)
            # Since we want smallest F values, we select based on -log(F), which gives largest values
            # This is equivalent to selecting dimensions with largest 1/F^p
            topk_k = min(self.topk, log_terms.size(1))  # Ensure k doesn't exceed total dimensions

            # Only perform topk if k < total dimensions (avoid unnecessary memory allocation)
            if topk_k < log_terms.size(1):
                # For each sample, select top-k dimensions with largest contribution (largest log_terms)
                topk_values, topk_indices = torch.topk(log_terms, k=topk_k, dim=1)  # [batch, topk_k]
            else:
                # If k >= total dimensions, use all dimensions (no selection needed)
                topk_values = log_terms
                topk_indices = torch.arange(log_terms.size(1), device=log_terms.device).unsqueeze(0).expand(log_terms.size(0), -1)

            # Collect gradient statistics if requested
            if self.collect_gradient_stats:
                # Create boolean mask for selected parameters (memory efficient)
                # topk_indices shape: [batch, topk_k]
                total_params = log_terms.size(1) if not self.use_topk else g_batch.size(1)
                selected_mask = torch.zeros(total_params, dtype=torch.bool, device=g_batch.device)
                selected_mask[topk_indices.flatten()] = True

                # Compute average gradient magnitude of selected parameters
                # g_batch shape: [batch, dim], topk_indices: [batch, topk_k]
                selected_gradients = torch.gather(g_batch, 1, topk_indices)  # [batch, topk_k]
                avg_gradient = torch.mean(torch.abs(selected_gradients)).item()

                self.gradient_stats.append({
                    'selected_mask': selected_mask.cpu(),  # Store boolean mask (CPU to save GPU memory)
                    'avg_gradient': avg_gradient
                })

            log_terms = topk_values  # [batch, topk_k]

        # LogSumExp for each sample
        max_log = torch.max(log_terms, dim=1, keepdim=True)[0]  # [batch, 1]
        log_energy = max_log.squeeze(1) + torch.log(
            torch.sum(torch.exp(log_terms - max_log), dim=1)
        )  # [batch]

        # Final score (higher = more OOD)
        # For entropy and KL gradients: reverse the sign because they behave oppositely
        # - NLL: ID (confident) → small gradient → small energy → need negation → high score for ID
        # - Entropy: ID (low H) → small gradient → small energy → no negation → low score for ID (opposite!)
        # - So for entropy: return positive log_energy to make ID → low score, OOD → high score
        if self.test_gradient_type in ["entropy", "kl"]:
            conf = log_energy.cpu()  # ID (small energy) → low score, OOD (large energy) → high score
        else:
            conf = -log_energy.cpu()  # ID (small energy) → high score, OOD (large energy) → low score, then negate

        return pred, conf

    def inference(self, net, data_loader, progress=True):
        """Run inference on data_loader."""
        pred_list, conf_list, label_list = super().inference(net, data_loader, progress)
        return pred_list, conf_list, label_list

    def set_hyperparam(self, hyperparam: list):
        """Set hyperparameters for APS mode.

        Args:
            hyperparam: List containing parameters in order:
                - [fisher_power] (base case)
                - [fisher_power, adaptive_alpha] (if use_adaptive_power=True)
                - [fisher_power, topk] (if use_topk=True, not adaptive)
                - [fisher_power, adaptive_alpha, topk] (if both enabled)
        """
        self.fisher_power = hyperparam[0]
        idx = 1

        # Handle adaptive_alpha if enabled
        if self.use_adaptive_power and len(hyperparam) > idx:
            self.adaptive_alpha = hyperparam[idx]
            idx += 1

        # Handle topk if enabled
        if self.use_topk and len(hyperparam) > idx:
            self.topk = int(hyperparam[idx])

    def get_hyperparam(self):
        """Get current hyperparameters for APS mode.

        Returns:
            List containing parameters based on enabled features:
                - [fisher_power] (base case)
                - [fisher_power, adaptive_alpha] (if use_adaptive_power=True)
                - [fisher_power, topk] (if use_topk=True, not adaptive)
                - [fisher_power, adaptive_alpha, topk] (if both enabled)
        """
        params = [self.fisher_power]

        if self.use_adaptive_power:
            params.append(self.adaptive_alpha)

        if self.use_topk:
            params.append(self.topk)

        return params
