from typing import Any
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.func import grad
except ImportError:
    from functorch import grad

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


class UnlearnPostprocessor(BasePostprocessor):
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

                # Compute logits and probabilities
                logits = F.linear(features, fc_weight, fc_bias if has_bias else None)
                probs = F.softmax(logits, dim=-1)

                # Compute gradient of NLL w.r.t. logits: ∇_z L = p - e_y
                # where p is the probability vector and e_y is one-hot label
                grad_logits = probs.clone()  # [batch_size, num_classes]
                grad_logits[torch.arange(len(labels)), labels] -= 1.0

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

    def _compute_fisher_energy(self, feature, fc_weight, fc_bias, target):
        """Compute Fisher-weighted energy: S(x) = g(x)^T F^{-p} g(x).

        Uses log-space computation for numerical stability with high powers.

        Args:
            feature: Sample feature [feature_dim]
            fc_weight: FC layer weight [num_classes, feature_dim]
            fc_bias: FC layer bias [num_classes] or None
            target: Pseudo-label target [num_classes] (for NLL loss)

        Returns:
            log_energy: log(Fisher energy) for numerical stability
        """
        eps_g = 1e-10  # Epsilon for gradients
        eps_f = 1e-8   # Epsilon for Fisher values

        # Compute gradient g(x) = ∇_θ [-log p(y*|x)] (NLL loss)
        if fc_bias is not None:
            def loss_fn(params):
                W_inner, b_inner = params
                out = F.linear(feature, W_inner, b_inner)
                log_probs = F.log_softmax(out, dim=-1)
                hard = target.argmax(dim=-1, keepdim=True)
                loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                return loss

            grads = grad(loss_fn)((fc_weight, fc_bias))
            grad_W, grad_b = grads[0], grads[1]
            g_x = torch.cat([grad_W.flatten(), grad_b.flatten()])

            # Get Fisher matrix (global)
            fisher_W = self.fisher_W_tensor
            fisher_b = self.fisher_b_tensor
            F_vec = torch.cat([fisher_W.flatten(), fisher_b.flatten()])
        else:
            def loss_fn(W_inner):
                out = F.linear(feature, W_inner)
                log_probs = F.log_softmax(out, dim=-1)
                hard = target.argmax(dim=-1, keepdim=True)
                loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                return loss

            grad_W = grad(loss_fn)(fc_weight)
            g_x = grad_W.flatten()

            # Get Fisher matrix (global)
            fisher_W = self.fisher_W_tensor
            F_vec = fisher_W.flatten()

        # Compute log(S(x)) = log(sum(g^2 / F^p)) using LogSumExp for stability
        # log(g^2 / F^p) = 2*log|g| - p*log F
        log_g_squared = 2.0 * torch.log(torch.abs(g_x) + eps_g)
        log_F_powered = self.fisher_power * torch.log(F_vec + eps_f)
        log_terms = log_g_squared - log_F_powered

        # LogSumExp trick: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        max_log = torch.max(log_terms)
        log_energy = max_log + torch.log(torch.sum(torch.exp(log_terms - max_log)))

        return log_energy

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

        # Get FC layer parameters
        fc_weight = fc.weight.data.clone()
        fc_bias = fc.bias.data.clone() if fc.bias is not None else None

        # Predictions
        pred = logits.argmax(dim=1).cpu()

        # Compute pseudo-labels (soft targets with temperature=1.0)
        targets = F.softmax(logits, dim=-1)

        # Compute Fisher energy for each sample
        conf_list = []
        for i in range(len(features)):
            log_fisher_energy = self._compute_fisher_energy(features[i], fc_weight, fc_bias, targets[i])
            # Negated log-energy (higher = more OOD)
            conf_list.append(-log_fisher_energy)

        conf = torch.stack(conf_list).cpu()

        return pred, conf
