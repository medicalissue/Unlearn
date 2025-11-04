from typing import Any
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.func import vmap, grad
except ImportError:
    from functorch import vmap, grad

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


class UnlearnPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.num_classes = num_classes_dict[self.config.dataset.name]

        # Core hyperparameters
        self.unlearn_mode = str(self.args.get("unlearn_mode", "ascent"))
        self.eta = float(self.args.get("eta", 1e-2))
        self.num_steps = int(self.args.get("num_steps", 100))
        self.temp = float(self.args.get("temp", 1.0))
        self.score_type = str(self.args.get("score_type", "prototype_coupling"))

        # Prototype coupling config
        pc_cfg = self.args.get("prototype_coupling_config", {}) or {}
        self.use_all_prototypes = bool(pc_cfg.get("use_all_prototypes", False))
        self.top_k = int(pc_cfg.get("top_k", 5))
        self.eigenvalue_mode = str(pc_cfg.get("eigenvalue_mode", "participation_ratio"))
        self.prototype_aggregation = str(pc_cfg.get("prototype_aggregation", "median"))  # "mean" or "median"
        self.fisher_mode = str(pc_cfg.get("fisher_mode", "classwise"))  # "global" or "classwise"

        # Max samples per class for prototype computation
        max_samples = pc_cfg.get("max_samples_per_class", None)
        if max_samples == "full" or max_samples is None:
            self.max_samples_per_class = None  # Use all samples
        else:
            self.max_samples_per_class = int(max_samples)

        # Will be set in setup()
        self.prototypes = None

        # Class-conditional statistics for normalized mode (will be set in setup())
        self.class_means = None  # [num_classes] - mean log(trace(C)) per class
        self.class_stds = None   # [num_classes] - std log(trace(C)) per class

        # Fisher Information Matrix for fisher mode (will be set in setup())
        # For "classwise" mode: stored as tensors [num_classes, ...]
        # For "global" mode: stored as single matrices
        self.fisher_W_tensor = None  # [num_classes, num_classes, feature_dim] for classwise, [num_classes, feature_dim] for global
        self.fisher_b_tensor = None  # [num_classes, num_classes] for classwise, [num_classes] for global, or None

    @staticmethod
    def _energy(logits: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(logits, dim=-1)

    def _pseudo_target(self, logit_row: torch.Tensor) -> torch.Tensor:
        """Compute soft pseudo-label target with temperature."""
        return F.softmax(logit_row / self.temp, dim=-1)

    def _get_prototype_cache_path(self, data_root: str) -> str:
        """Get the path to the cached prototype file."""
        cache_dir = os.path.join(data_root, 'prototypes')
        os.makedirs(cache_dir, exist_ok=True)
        # Include sample limit in filename (different limits = different prototypes)
        if self.max_samples_per_class is None:
            sample_suffix = "full"
        else:
            sample_suffix = f"samples{self.max_samples_per_class}"
        filename = f"{self.config.dataset.name}_prototypes_{self.prototype_aggregation}_{sample_suffix}.pt"
        return os.path.join(cache_dir, filename)

    def _get_class_stats_cache_path(self, data_root: str) -> str:
        """Get the path to the cached class statistics file."""
        cache_dir = os.path.join(data_root, 'prototypes')
        os.makedirs(cache_dir, exist_ok=True)
        filename = f"{self.config.dataset.name}_class_stats_eta{self.eta}_steps{self.num_steps}.pt"
        return os.path.join(cache_dir, filename)

    def _get_fisher_cache_path(self, data_root: str) -> str:
        """Get the path to the cached Fisher matrices file."""
        cache_dir = os.path.join(data_root, 'prototypes')
        os.makedirs(cache_dir, exist_ok=True)
        filename = f"{self.config.dataset.name}_fisher_matrices.pt"
        return os.path.join(cache_dir, filename)

    def _save_prototypes(self, path: str):
        """Save prototypes to disk."""
        torch.save({
            'prototypes': self.prototypes.cpu(),
            'dataset_name': self.config.dataset.name,
            'num_classes': self.num_classes,
            'aggregation_method': self.prototype_aggregation,
            'max_samples_per_class': self.max_samples_per_class,
        }, path)
        print(f"✓ Saved prototypes to {path}")

    def _load_prototypes(self, path: str) -> torch.Tensor:
        """Load prototypes from disk."""
        checkpoint = torch.load(path)
        # Verify metadata
        assert checkpoint['dataset_name'] == self.config.dataset.name, \
            f"Dataset mismatch: {checkpoint['dataset_name']} vs {self.config.dataset.name}"
        assert checkpoint['num_classes'] == self.num_classes, \
            f"Number of classes mismatch: {checkpoint['num_classes']} vs {self.num_classes}"
        assert checkpoint['aggregation_method'] == self.prototype_aggregation, \
            f"Aggregation method mismatch: {checkpoint['aggregation_method']} vs {self.prototype_aggregation}"
        assert checkpoint.get('max_samples_per_class', None) == self.max_samples_per_class, \
            f"Sample limit mismatch: {checkpoint.get('max_samples_per_class', None)} vs {self.max_samples_per_class}"
        return checkpoint['prototypes'].cuda()

    def _save_class_stats(self, path: str):
        """Save class statistics to disk."""
        torch.save({
            'class_means': self.class_means.cpu(),
            'class_stds': self.class_stds.cpu(),
            'dataset_name': self.config.dataset.name,
            'num_classes': self.num_classes,
            'eta': self.eta,
            'num_steps': self.num_steps,
        }, path)
        print(f"✓ Saved class statistics to {path}")

    def _load_class_stats(self, path: str):
        """Load class statistics from disk."""
        checkpoint = torch.load(path)
        # Verify metadata
        assert checkpoint['dataset_name'] == self.config.dataset.name, \
            f"Dataset mismatch: {checkpoint['dataset_name']} vs {self.config.dataset.name}"
        assert checkpoint['num_classes'] == self.num_classes, \
            f"Number of classes mismatch: {checkpoint['num_classes']} vs {self.num_classes}"
        assert checkpoint['eta'] == self.eta, \
            f"eta mismatch: {checkpoint['eta']} vs {self.eta}"
        assert checkpoint['num_steps'] == self.num_steps, \
            f"num_steps mismatch: {checkpoint['num_steps']} vs {self.num_steps}"
        self.class_means = checkpoint['class_means'].cuda()
        self.class_stds = checkpoint['class_stds'].cuda()
        print(f"✓ Loaded class statistics from {path}")

    def _save_fisher_matrices(self, path: str):
        """Save Fisher matrices to disk."""
        torch.save({
            'fisher_W_tensor': self.fisher_W_tensor.cpu(),
            'fisher_b_tensor': self.fisher_b_tensor.cpu() if self.fisher_b_tensor is not None else None,
            'dataset_name': self.config.dataset.name,
            'num_classes': self.num_classes,
        }, path)
        print(f"✓ Saved Fisher matrices to {path}")

    def _load_fisher_matrices(self, path: str):
        """Load Fisher matrices from disk."""
        checkpoint = torch.load(path)
        # Verify metadata
        assert checkpoint['dataset_name'] == self.config.dataset.name, \
            f"Dataset mismatch: {checkpoint['dataset_name']} vs {self.config.dataset.name}"
        assert checkpoint['num_classes'] == self.num_classes, \
            f"Number of classes mismatch: {checkpoint['num_classes']} vs {self.num_classes}"

        self.fisher_W_tensor = checkpoint['fisher_W_tensor'].cuda()
        self.fisher_b_tensor = checkpoint['fisher_b_tensor'].cuda() if checkpoint['fisher_b_tensor'] is not None else None
        print(f"✓ Loaded Fisher matrices from {path}")

    def _get_fc_params(self, net):
        """Extract final FC layer parameters."""
        fc = net.fc if hasattr(net, 'fc') else net.module.fc if hasattr(net, 'module') else None
        if fc is None:
            raise ValueError("Cannot find FC layer")
        return fc.weight.data.clone(), fc.bias.data.clone()

    def _compute_fisher_energy(self, feature, fc_weight, fc_bias, target):
        """Compute Fisher-weighted unlearning energy: S(x) = g(x)^T F^{-1} g(x).

        Args:
            feature: Sample feature [feature_dim]
            fc_weight: FC layer weight [num_classes, feature_dim]
            fc_bias: FC layer bias [num_classes] or None
            target: Pseudo-label target [num_classes]

        Returns:
            energy: Fisher-weighted energy (scalar)
        """
        # Compute gradient g(x) = ∇_θ ℓ(x; θ)
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

            # Flatten gradients into vector g(x)
            g_x = torch.cat([grad_W.flatten(), grad_b.flatten()])  # [D]

            # Get Fisher matrix
            if self.fisher_mode == "global":
                fisher_W = self.fisher_W_tensor
                fisher_b = self.fisher_b_tensor
            else:  # classwise
                # Select Fisher based on predicted class (vmap-safe)
                logits = F.linear(feature, fc_weight, fc_bias)
                pred_class_idx = logits.argmax(dim=-1)
                one_hot_pred = F.one_hot(pred_class_idx, num_classes=self.num_classes).float()
                fisher_W = torch.sum(one_hot_pred.view(-1, 1, 1) * self.fisher_W_tensor, dim=0)
                fisher_b = torch.sum(one_hot_pred * self.fisher_b_tensor, dim=0)

            # Flatten Fisher into vector F
            F_vec = torch.cat([fisher_W.flatten(), fisher_b.flatten()])  # [D]

        else:
            def loss_fn(W_inner):
                out = F.linear(feature, W_inner)
                log_probs = F.log_softmax(out, dim=-1)
                hard = target.argmax(dim=-1, keepdim=True)
                loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                return loss

            grad_W = grad(loss_fn)(fc_weight)
            g_x = grad_W.flatten()  # [D]

            # Get Fisher matrix
            if self.fisher_mode == "global":
                fisher_W = self.fisher_W_tensor
            else:  # classwise
                logits = F.linear(feature, fc_weight)
                pred_class_idx = logits.argmax(dim=-1)
                one_hot_pred = F.one_hot(pred_class_idx, num_classes=self.num_classes).float()
                fisher_W = torch.sum(one_hot_pred.view(-1, 1, 1) * self.fisher_W_tensor, dim=0)

            F_vec = fisher_W.flatten()  # [D]

        # Compute S(x) = g(x)^T F^{-1} g(x)
        # Use element-wise division as approximation: g^T F^{-1} g ≈ sum(g^2 / F)
        # This avoids expensive matrix inversion and is equivalent when F is diagonal
        energy = torch.sum((g_x ** 2) / (F_vec + 1e-8))

        return energy

    def _compute_energy_at_step(self, W_base, b_base, W_current, b_current,
                                prototype_tensor, logits_orig, device):
        """Compute log(trace(C)) energy at current unlearning step.

        This is a lightweight version of _compute_prototype_coupling_score
        that only returns the log(trace(C)) value without normalization.

        Returns:
            energy: log(trace(C)) as a scalar tensor
        """
        num_classes = prototype_tensor.shape[0]

        # Select which prototypes to use
        if self.use_all_prototypes:
            proto_indices = torch.arange(num_classes, device=device)
        else:
            _, top_k_indices = torch.topk(logits_orig, k=min(self.top_k, num_classes), dim=-1)
            proto_indices = top_k_indices

        # Collect gradient vectors
        grad_vecs_base = []
        grad_vecs_current = []

        for i in range(proto_indices.shape[0]):
            idx = proto_indices[i]
            one_hot_i = F.one_hot(idx, num_classes=num_classes).float()
            proto_feature_i = torch.matmul(one_hot_i, prototype_tensor)

            # Base gradients
            if b_base is not None:
                def proto_loss_i_base(params):
                    W_inner, b_inner = params
                    out = F.linear(proto_feature_i, W_inner, b_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = idx.unsqueeze(-1)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss
                grads_i_base = grad(proto_loss_i_base)((W_base, b_base))
                grad_vec_i_base = torch.cat([grads_i_base[0].flatten(), grads_i_base[1].flatten()])
            else:
                def proto_loss_i_base(W_inner):
                    out = F.linear(proto_feature_i, W_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = idx.unsqueeze(-1)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss
                grad_W_i_base = grad(proto_loss_i_base)(W_base)
                grad_vec_i_base = grad_W_i_base.flatten()

            # Current gradients
            if b_current is not None:
                def proto_loss_i_current(params):
                    W_inner, b_inner = params
                    out = F.linear(proto_feature_i, W_inner, b_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = idx.unsqueeze(-1)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss
                grads_i_current = grad(proto_loss_i_current)((W_current, b_current))
                grad_vec_i_current = torch.cat([grads_i_current[0].flatten(), grads_i_current[1].flatten()])
            else:
                def proto_loss_i_current(W_inner):
                    out = F.linear(proto_feature_i, W_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = idx.unsqueeze(-1)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss
                grad_W_i_current = grad(proto_loss_i_current)(W_current)
                grad_vec_i_current = grad_W_i_current.flatten()

            grad_vecs_base.append(grad_vec_i_base)
            grad_vecs_current.append(grad_vec_i_current)

        # Compute coupling matrix
        grad_vecs_base = torch.stack(grad_vecs_base)
        grad_vecs_current = torch.stack(grad_vecs_current)
        delta_grads = grad_vecs_current - grad_vecs_base
        coupling_matrix = torch.matmul(delta_grads, delta_grads.T)
        trace_C = torch.trace(coupling_matrix)

        # Return log(trace(C))
        return torch.log(trace_C + 1e-10)

    def _compute_prototype_coupling_score(self, W_base, b_base, W_current, b_current,
                                         prototype_tensor, logits_orig, device):
        """Compute prototype coupling score using full gradient coupling matrix.

        This computes gradients for each selected prototype and builds a coupling matrix
        to measure how prototypes are affected by the unlearning process.

        ID samples: LOW PR (strongly coupled to specific prototype) → HIGH score (negated)
        OOD samples: HIGH PR (weakly coupled to many prototypes) → LOW score (negated)
        """
        num_classes = prototype_tensor.shape[0]

        # Select which prototypes to use
        if self.use_all_prototypes:
            proto_indices = torch.arange(num_classes, device=device)
        else:
            _, top_k_indices = torch.topk(logits_orig, k=min(self.top_k, num_classes), dim=-1)
            proto_indices = top_k_indices

        # Collect FULL gradient vectors for all selected prototypes (before and after)
        grad_vecs_base = []
        grad_vecs_current = []

        for i in range(proto_indices.shape[0]):
            # Get prototype index (vmap-safe: use one-hot indexing)
            idx = proto_indices[i]
            one_hot_i = F.one_hot(idx, num_classes=num_classes).float()
            proto_feature_i = torch.matmul(one_hot_i, prototype_tensor)

            # Compute FULL gradient on BASE FC with this prototype
            if b_base is not None:
                def proto_loss_i_base(params):
                    W_inner, b_inner = params
                    out = F.linear(proto_feature_i, W_inner, b_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = idx.unsqueeze(-1)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss
                grads_i_base = grad(proto_loss_i_base)((W_base, b_base))
                grad_vec_i_base = torch.cat([grads_i_base[0].flatten(), grads_i_base[1].flatten()])
            else:
                def proto_loss_i_base(W_inner):
                    out = F.linear(proto_feature_i, W_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = idx.unsqueeze(-1)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss
                grad_W_i_base = grad(proto_loss_i_base)(W_base)
                grad_vec_i_base = grad_W_i_base.flatten()

            # Compute FULL gradient on CURRENT FC with this prototype
            if b_current is not None:
                def proto_loss_i_current(params):
                    W_inner, b_inner = params
                    out = F.linear(proto_feature_i, W_inner, b_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = idx.unsqueeze(-1)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss
                grads_i_current = grad(proto_loss_i_current)((W_current, b_current))
                grad_vec_i_current = torch.cat([grads_i_current[0].flatten(), grads_i_current[1].flatten()])
            else:
                def proto_loss_i_current(W_inner):
                    out = F.linear(proto_feature_i, W_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = idx.unsqueeze(-1)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss
                grad_W_i_current = grad(proto_loss_i_current)(W_current)
                grad_vec_i_current = grad_W_i_current.flatten()

            grad_vecs_base.append(grad_vec_i_base)
            grad_vecs_current.append(grad_vec_i_current)

        # Stack into matrices [num_prototypes, param_size]
        grad_vecs_base = torch.stack(grad_vecs_base)  # [K, D]
        grad_vecs_current = torch.stack(grad_vecs_current)  # [K, D]

        # Compute gradient change vectors [num_prototypes, param_size]
        delta_grads = grad_vecs_current - grad_vecs_base  # [K, D]

        # Build coupling matrix: C = delta_grads @ delta_grads.T [K, K]
        coupling_matrix = torch.matmul(delta_grads, delta_grads.T)  # [K, K]

        # Compute trace of coupling matrix
        trace_C = torch.trace(coupling_matrix)

        # Choose eigenvalue-based metric
        if self.eigenvalue_mode == "log_trace":
            # log(trace(C)) - measures total variance/energy
            # ID samples: HIGH trace → HIGH score (negated)
            # OOD samples: LOW trace → LOW score (negated)
            score = -torch.log(trace_C + 1e-10)
        elif self.eigenvalue_mode == "log_trace_normalized":
            # Class-conditional normalized log(trace(C))
            # Compute raw log(trace(C))
            log_trace_energy = torch.log(trace_C + 1e-10)

            # Get predicted class from original logits
            y_pred = logits_orig.argmax(dim=-1)

            # Get class statistics using one-hot indexing (vmap-safe)
            one_hot_pred = F.one_hot(y_pred, num_classes=self.num_classes).float()
            class_mean = torch.matmul(one_hot_pred, self.class_means)
            class_std = torch.matmul(one_hot_pred, self.class_stds)

            # Compute z-score
            z_score = (log_trace_energy - class_mean) / (class_std + 1e-8)

            # OOD score: -|z_score| (ID samples have z ≈ 0, OOD samples have large |z|)
            score = -torch.abs(z_score)
        else:  # "participation_ratio" (default)
            # Compute participation ratio: PR = (trace(C))^2 / trace(C^2)
            trace_C2 = torch.trace(torch.matmul(coupling_matrix, coupling_matrix))
            pr = (trace_C ** 2) / (trace_C2 + 1e-10)
            # Return negated score (lower PR = ID → higher score)
            score = pr

        return score

    def _single_sample_unlearn(self, fc_weight, fc_bias, feature, prototype_tensor,
                              global_proto, device):
        """Unlearn a single sample using gradient ascent.

        Args:
            fc_weight: FC layer weight [num_classes, feature_dim]
            fc_bias: FC layer bias [num_classes]
            feature: Single sample feature [feature_dim]
            prototype_tensor: Class prototypes [num_classes, feature_dim]
            global_proto: Global prototype [feature_dim] (unused for now)
            device: torch device

        Returns:
            score: OOD score (scalar)
        """
        # Store base parameters
        W_base = fc_weight
        b_base = fc_bias

        # Initialize working parameters
        W = fc_weight.clone()
        b = fc_bias.clone()

        # Compute initial logits
        if b is not None:
            logits_init = F.linear(feature, W, b)
        else:
            logits_init = F.linear(feature, W)

        # Compute pseudo target once
        target = self._pseudo_target(logits_init).to(device=device, dtype=logits_init.dtype)

        # Track initial energy for energy_growth_rate mode
        if self.score_type == "energy_growth_rate":
            energy_0 = self._compute_energy_at_step(W_base, b_base, W, b, prototype_tensor, logits_init, device)

        # Unlearning loop
        for step in range(self.num_steps):
            # Compute loss gradient for gradient ascent
            if b is not None:
                def loss_fn(params):
                    W_inner, b_inner = params
                    out = F.linear(feature, W_inner, b_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = target.argmax(dim=-1, keepdim=True)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss

                grads = grad(loss_fn)((W, b))
                grad_W, grad_b = grads[0], grads[1]

                # Update based on unlearn_mode
                if self.unlearn_mode == "fisher":
                    if self.fisher_mode == "global":
                        # Global Fisher mode: use global Fisher matrix (no class selection)
                        fisher_W_selected = self.fisher_W_tensor
                        fisher_b_selected = self.fisher_b_tensor
                    else:  # classwise mode
                        # Fisher-weighted update: use predicted class's Fisher matrix (vmap-safe)
                        pred_class_idx = logits_init.argmax(dim=-1)
                        one_hot_pred = F.one_hot(pred_class_idx, num_classes=self.num_classes).float()

                        # Select Fisher matrix using one-hot (weighted sum across classes)
                        # fisher_W/b are [num_classes, ...], one_hot_pred is [num_classes]
                        fisher_W_selected = torch.sum(one_hot_pred.view(-1, 1, 1) * self.fisher_W_tensor, dim=0)
                        fisher_b_selected = torch.sum(one_hot_pred * self.fisher_b_tensor, dim=0) if self.fisher_b_tensor is not None else None

                    # Fisher-weighted gradient ascent
                    W = W + self.eta * (fisher_W_selected * grad_W)
                    b = b + self.eta * (fisher_b_selected * grad_b)
                else:  # "ascent"
                    # Standard gradient ascent update
                    W = W + self.eta * grad_W
                    b = b + self.eta * grad_b
            else:
                def loss_fn(W_inner):
                    out = F.linear(feature, W_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = target.argmax(dim=-1, keepdim=True)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss

                grad_W = grad(loss_fn)(W)

                # Update based on unlearn_mode
                if self.unlearn_mode == "fisher":
                    if self.fisher_mode == "global":
                        # Global Fisher mode: use global Fisher matrix (no class selection)
                        fisher_W_selected = self.fisher_W_tensor
                    else:  # classwise mode
                        # Fisher-weighted update (vmap-safe)
                        pred_class_idx = logits_init.argmax(dim=-1)
                        one_hot_pred = F.one_hot(pred_class_idx, num_classes=self.num_classes).float()

                        # Select Fisher matrix using one-hot
                        fisher_W_selected = torch.sum(one_hot_pred.view(-1, 1, 1) * self.fisher_W_tensor, dim=0)

                    W = W + self.eta * (fisher_W_selected * grad_W)
                else:  # "ascent"
                    # Standard gradient ascent update
                    W = W + self.eta * grad_W

        # Compute OOD score
        if self.score_type == "prototype_coupling":
            score = self._compute_prototype_coupling_score(
                W_base, b_base, W, b, prototype_tensor, logits_init, device
            )
        elif self.score_type == "energy_growth_rate":
            # Compute final energy and growth rate
            energy_T = self._compute_energy_at_step(W_base, b_base, W, b, prototype_tensor, logits_init, device)
            growth_rate = (energy_T - energy_0) / self.num_steps
            # ID samples: high growth rate → high score
            # OOD samples: low growth rate → low score
            score = -growth_rate
        elif self.score_type == "fisher_energy":
            # Compute Fisher-weighted unlearning energy: S(x) = g(x)^T F^{-1} g(x)
            # ID samples: low Fisher energy (gradient aligns with ID distribution)
            # OOD samples: high Fisher energy (gradient doesn't align with ID)
            fisher_energy = self._compute_fisher_energy(feature, W_base, b_base, target)
            # Return negated energy so higher score = more OOD
            score = -fisher_energy
        else:
            # Default: use energy change
            if b is not None:
                logits_final = F.linear(feature, W, b)
            else:
                logits_final = F.linear(feature, W)

            energy_init = self._energy(logits_init)
            energy_final = self._energy(logits_final)
            score = energy_final - energy_init

        return score

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """Compute class prototypes from ID training data."""
        import tqdm
        net.eval()

        # Determine data root path
        if hasattr(self.config.dataset, 'data_root') and self.config.dataset.data_root is not None:
            data_root = self.config.dataset.data_root
        else:
            data_root = './data'

        # Try to load cached prototypes
        cache_path = self._get_prototype_cache_path(data_root)
        prototypes_loaded = False
        if os.path.exists(cache_path):
            print(f"Loading cached prototypes from {cache_path}")
            self.prototypes = self._load_prototypes(cache_path)
            print(f"✓ Loaded {self.num_classes} class prototypes using {self.prototype_aggregation}")
            prototypes_loaded = True

        # Only compute prototypes if not loaded from cache
        if not prototypes_loaded:
            # Use train data if available, otherwise fall back to val data
            if 'train' in id_loader_dict:
                data_loader = id_loader_dict['train']
            elif 'val' in id_loader_dict:
                print("Warning: Training data not available, using validation data for prototype computation")
                data_loader = id_loader_dict['val']
            else:
                raise ValueError("Neither training nor validation data available for setup")

            # Accumulate features per class with sample limiting during extraction
            if self.max_samples_per_class is None:
                # Extract all features (original behavior)
                all_features = []
                all_labels = []

                print("Computing class prototypes...")
                with torch.no_grad():
                    for batch in tqdm.tqdm(data_loader, desc="Extracting features"):
                        data = batch['data'].cuda()
                        labels = batch['label'].cuda()

                        _, features = net(data, return_feature=True)

                        all_features.append(features)
                        all_labels.append(labels)

                # Concatenate all features and labels
                all_features = torch.cat(all_features, dim=0)  # [N, feat_dim]
                all_labels = torch.cat(all_labels, dim=0)  # [N]
            else:
                # Extract limited samples per class (efficient for small max_samples_per_class)
                print(f"Computing class prototypes (extracting max {self.max_samples_per_class} samples per class)...")
                class_features = {c: [] for c in range(self.num_classes)}
                samples_per_class = {c: 0 for c in range(self.num_classes)}

                target_total = self.max_samples_per_class * self.num_classes
                pbar = tqdm.tqdm(total=target_total, desc="Extracting features")

                with torch.no_grad():
                    for batch in data_loader:
                        # Check if we've collected enough samples for all classes
                        if all(samples_per_class[c] >= self.max_samples_per_class for c in range(self.num_classes)):
                            break

                        data = batch['data'].cuda()
                        labels = batch['label'].cuda()

                        # Extract features for the entire batch
                        _, features = net(data, return_feature=True)

                        # Store features per class (only if needed)
                        for feat, lbl in zip(features, labels):
                            c = lbl.item()
                            if samples_per_class[c] < self.max_samples_per_class:
                                class_features[c].append(feat)
                                samples_per_class[c] += 1
                                pbar.update(1)

                pbar.close()

                # Convert to all_features format for compatibility with prototype computation
                all_features = []
                all_labels = []
                for c in range(self.num_classes):
                    if len(class_features[c]) > 0:
                        class_feats = torch.stack(class_features[c])
                        all_features.append(class_feats)
                        all_labels.append(torch.full((len(class_feats),), c, dtype=torch.long, device=class_feats.device))

                if len(all_features) > 0:
                    all_features = torch.cat(all_features, dim=0)
                    all_labels = torch.cat(all_labels, dim=0)
                else:
                    raise ValueError("No features extracted from dataset")

            # Compute prototypes using loop-based approach (memory efficient)
            # Note: if max_samples_per_class is set, samples are already limited during extraction
            if self.prototype_aggregation == "mean":
                print(f"Computing prototypes using mean...")
                prototypes = []
                for c in range(self.num_classes):
                    mask = (all_labels == c)
                    if mask.sum() > 0:
                        class_features = all_features[mask]
                        num_available = len(class_features)
                        print(f"  Class {c}: {num_available} samples")
                        prototype = class_features.mean(dim=0)
                    else:
                        print(f"  Warning: No samples found for class {c}")
                        prototype = torch.zeros(all_features.shape[1], device=all_features.device)
                    prototypes.append(prototype)
                self.prototypes = torch.stack(prototypes)
            else:
                # Median computation
                print(f"Computing prototypes using median...")
                prototypes = []
                for c in range(self.num_classes):
                    mask = (all_labels == c)
                    if mask.sum() > 0:
                        class_features = all_features[mask]
                        num_available = len(class_features)
                        print(f"  Class {c}: {num_available} samples")
                        prototype = torch.median(class_features, dim=0).values
                    else:
                        print(f"  Warning: No samples found for class {c}")
                        prototype = torch.zeros(all_features.shape[1], device=all_features.device)
                    prototypes.append(prototype)
                self.prototypes = torch.stack(prototypes)

            # Save prototypes for future use
            self._save_prototypes(cache_path)
            print(f"✓ Computed {self.num_classes} class prototypes using {self.prototype_aggregation}")

        # Compute class statistics for log_trace_normalized mode
        if self.eigenvalue_mode == "log_trace_normalized":
            stats_cache_path = self._get_class_stats_cache_path(data_root)

            # Try to load cached statistics
            if os.path.exists(stats_cache_path):
                print(f"Loading cached class statistics from {stats_cache_path}")
                self._load_class_stats(stats_cache_path)
                return

            print("Computing class-conditional statistics for log_trace_normalized mode...")

            # Use validation data for calibration
            if 'val' in id_loader_dict:
                val_loader = id_loader_dict['val']
            elif 'train' in id_loader_dict:
                print("Warning: Validation data not available, using training data for calibration")
                val_loader = id_loader_dict['train']
            else:
                raise ValueError("No data available for class statistics calibration")

            # Collect log(trace(C)) values per predicted class
            energies_per_class = {c: [] for c in range(self.num_classes)}

            # Get FC parameters
            fc_weight, fc_bias = self._get_fc_params(net)

            with torch.no_grad():
                for batch in tqdm.tqdm(val_loader, desc="Calibrating class statistics"):
                    data = batch['data'].cuda()

                    # Extract features and predictions
                    logits, features = net(data, return_feature=True)
                    predictions = logits.argmax(dim=1)

                    # Compute log(trace(C)) for each sample
                    for i in range(data.shape[0]):
                        feature = features[i]
                        y_pred = predictions[i].item()

                        # Perform unlearning to get log(trace(C))
                        W = fc_weight.clone()
                        b = fc_bias.clone() if fc_bias is not None else None

                        # Compute initial logits
                        if b is not None:
                            logits_init = F.linear(feature, W, b)
                        else:
                            logits_init = F.linear(feature, W)

                        # Compute pseudo target
                        target = self._pseudo_target(logits_init)

                        # Unlearning loop
                        for step in range(self.num_steps):
                            if b is not None:
                                def loss_fn(params):
                                    W_inner, b_inner = params
                                    out = F.linear(feature, W_inner, b_inner)
                                    log_probs = F.log_softmax(out, dim=-1)
                                    hard = target.argmax(dim=-1, keepdim=True)
                                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                                    return loss
                                grads = grad(loss_fn)((W, b))
                                W = W + self.eta * grads[0]
                                b = b + self.eta * grads[1]
                            else:
                                def loss_fn(W_inner):
                                    out = F.linear(feature, W_inner)
                                    log_probs = F.log_softmax(out, dim=-1)
                                    hard = target.argmax(dim=-1, keepdim=True)
                                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                                    return loss
                                grad_W = grad(loss_fn)(W)
                                W = W + self.eta * grad_W

                        # Compute coupling matrix and log(trace(C))
                        # Select prototypes
                        if self.use_all_prototypes:
                            proto_indices = torch.arange(self.num_classes, device=feature.device)
                        else:
                            _, top_k_indices = torch.topk(logits_init, k=min(self.top_k, self.num_classes))
                            proto_indices = top_k_indices

                        # Compute gradient vectors
                        grad_vecs_base = []
                        grad_vecs_current = []

                        for j in range(proto_indices.shape[0]):
                            idx = proto_indices[j]
                            one_hot_j = F.one_hot(idx, num_classes=self.num_classes).float()
                            proto_feature_j = torch.matmul(one_hot_j, self.prototypes)

                            # Base gradients
                            if fc_bias is not None:
                                def proto_loss_base(params):
                                    W_inner, b_inner = params
                                    out = F.linear(proto_feature_j, W_inner, b_inner)
                                    log_probs = F.log_softmax(out, dim=-1)
                                    hard = idx.unsqueeze(-1)
                                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                                    return loss
                                grads_base = grad(proto_loss_base)((fc_weight, fc_bias))
                                grad_vec_base = torch.cat([grads_base[0].flatten(), grads_base[1].flatten()])
                            else:
                                def proto_loss_base(W_inner):
                                    out = F.linear(proto_feature_j, W_inner)
                                    log_probs = F.log_softmax(out, dim=-1)
                                    hard = idx.unsqueeze(-1)
                                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                                    return loss
                                grad_W_base = grad(proto_loss_base)(fc_weight)
                                grad_vec_base = grad_W_base.flatten()

                            # Current gradients
                            if b is not None:
                                def proto_loss_current(params):
                                    W_inner, b_inner = params
                                    out = F.linear(proto_feature_j, W_inner, b_inner)
                                    log_probs = F.log_softmax(out, dim=-1)
                                    hard = idx.unsqueeze(-1)
                                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                                    return loss
                                grads_current = grad(proto_loss_current)((W, b))
                                grad_vec_current = torch.cat([grads_current[0].flatten(), grads_current[1].flatten()])
                            else:
                                def proto_loss_current(W_inner):
                                    out = F.linear(proto_feature_j, W_inner)
                                    log_probs = F.log_softmax(out, dim=-1)
                                    hard = idx.unsqueeze(-1)
                                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                                    return loss
                                grad_W_current = grad(proto_loss_current)(W)
                                grad_vec_current = grad_W_current.flatten()

                            grad_vecs_base.append(grad_vec_base)
                            grad_vecs_current.append(grad_vec_current)

                        # Compute coupling matrix
                        grad_vecs_base = torch.stack(grad_vecs_base)
                        grad_vecs_current = torch.stack(grad_vecs_current)
                        delta_grads = grad_vecs_current - grad_vecs_base
                        coupling_matrix = torch.matmul(delta_grads, delta_grads.T)
                        trace_C = torch.trace(coupling_matrix)

                        # Store log(trace(C))
                        log_trace_energy = torch.log(trace_C + 1e-10).item()
                        energies_per_class[y_pred].append(log_trace_energy)

            # Compute mean and std per class
            class_means = []
            class_stds = []

            for c in range(self.num_classes):
                if len(energies_per_class[c]) > 0:
                    energies_tensor = torch.tensor(energies_per_class[c])
                    mean_c = energies_tensor.mean().item()
                    std_c = energies_tensor.std().item() if len(energies_per_class[c]) > 1 else 1.0
                    print(f"  Class {c}: mean={mean_c:.4f}, std={std_c:.4f}, n={len(energies_per_class[c])}")
                else:
                    print(f"  Warning: No samples for class {c}, using default statistics")
                    mean_c = 0.0
                    std_c = 1.0

                class_means.append(mean_c)
                class_stds.append(std_c)

            self.class_means = torch.tensor(class_means, dtype=torch.float32).cuda()
            self.class_stds = torch.tensor(class_stds, dtype=torch.float32).cuda()

            # Save statistics
            self._save_class_stats(stats_cache_path)
            print(f"✓ Computed class-conditional statistics for {self.num_classes} classes")

        # Compute Fisher Information Matrix for fisher mode
        if self.unlearn_mode == "fisher":
            fisher_cache_path = self._get_fisher_cache_path(data_root)

            # Try to load cached Fisher matrices
            if os.path.exists(fisher_cache_path):
                print(f"Loading cached Fisher matrices from {fisher_cache_path}")
                self._load_fisher_matrices(fisher_cache_path)
                return

            if self.fisher_mode == "global":
                print("Computing global Fisher Information Matrix for fisher mode...")
            else:
                print("Computing Fisher Information Matrix per class for fisher mode...")

            # Use training data for Fisher computation
            if 'train' in id_loader_dict:
                train_loader = id_loader_dict['train']
            elif 'val' in id_loader_dict:
                print("Warning: Training data not available, using validation data for Fisher computation")
                train_loader = id_loader_dict['val']
            else:
                raise ValueError("No data available for Fisher computation")

            # Get FC parameters
            fc_weight, fc_bias = self._get_fc_params(net)

            if self.fisher_mode == "global":
                # Global mode: accumulate across all samples regardless of class
                fisher_W_global = torch.zeros_like(fc_weight)
                fisher_b_global = torch.zeros_like(fc_bias) if fc_bias is not None else None
                total_count = 0

                with torch.no_grad():
                    for batch in tqdm.tqdm(train_loader, desc="Computing global Fisher matrix"):
                        data = batch['data'].cuda()
                        labels = batch['label'].cuda()

                        # Extract features and compute logits
                        logits, features = net(data, return_feature=True)

                        # Process each sample
                        for i in range(data.shape[0]):
                            feature = features[i]
                            label = labels[i].item()

                            # Compute gradient of negative log-likelihood w.r.t. FC parameters
                            if fc_bias is not None:
                                def nll_loss(params):
                                    W_inner, b_inner = params
                                    out = F.linear(feature, W_inner, b_inner)
                                    log_probs = F.log_softmax(out, dim=-1)
                                    loss = -log_probs[label]
                                    return loss

                                grads = grad(nll_loss)((fc_weight, fc_bias))
                                grad_W, grad_b = grads[0], grads[1]

                                # Accumulate squared gradients globally
                                fisher_W_global += grad_W ** 2
                                fisher_b_global += grad_b ** 2
                            else:
                                def nll_loss(W_inner):
                                    out = F.linear(feature, W_inner)
                                    log_probs = F.log_softmax(out, dim=-1)
                                    loss = -log_probs[label]
                                    return loss

                                grad_W = grad(nll_loss)(fc_weight)
                                fisher_W_global += grad_W ** 2

                            total_count += 1

                # Average to get global Fisher matrix
                fisher_W_global = fisher_W_global / total_count
                fisher_b_global = fisher_b_global / total_count if fc_bias is not None else None

                # Store global Fisher (no per-class dimension needed)
                self.fisher_W_tensor = fisher_W_global
                self.fisher_b_tensor = fisher_b_global

                print(f"  Global Fisher: {total_count} samples, Fisher norm W={fisher_W_global.norm().item():.4f}")
                print(f"✓ Computed global Fisher matrix")

            else:  # classwise mode
                # Accumulate squared gradients per class
                fisher_W_acc = {c: torch.zeros_like(fc_weight) for c in range(self.num_classes)}
                fisher_b_acc = {c: torch.zeros_like(fc_bias) if fc_bias is not None else None for c in range(self.num_classes)}
                class_counts = {c: 0 for c in range(self.num_classes)}

                with torch.no_grad():
                    for batch in tqdm.tqdm(train_loader, desc="Computing Fisher matrices"):
                        data = batch['data'].cuda()
                        labels = batch['label'].cuda()

                        # Extract features and compute logits
                        logits, features = net(data, return_feature=True)

                        # Process each sample
                        for i in range(data.shape[0]):
                            feature = features[i]
                            label = labels[i].item()

                            # Compute gradient of negative log-likelihood w.r.t. FC parameters
                            # ∇_θ [-log p(y|x)] for the true class
                            if fc_bias is not None:
                                def nll_loss(params):
                                    W_inner, b_inner = params
                                    out = F.linear(feature, W_inner, b_inner)
                                    log_probs = F.log_softmax(out, dim=-1)
                                    # Negative log-likelihood for true label
                                    loss = -log_probs[label]
                                    return loss

                                grads = grad(nll_loss)((fc_weight, fc_bias))
                                grad_W, grad_b = grads[0], grads[1]

                                # Accumulate squared gradients (Fisher approximation)
                                fisher_W_acc[label] += grad_W ** 2
                                fisher_b_acc[label] += grad_b ** 2
                            else:
                                def nll_loss(W_inner):
                                    out = F.linear(feature, W_inner)
                                    log_probs = F.log_softmax(out, dim=-1)
                                    loss = -log_probs[label]
                                    return loss

                                grad_W = grad(nll_loss)(fc_weight)
                                fisher_W_acc[label] += grad_W ** 2

                            class_counts[label] += 1

                # Average and stack to get Fisher tensors [num_classes, ...]
                fisher_W_list = []
                fisher_b_list = []

                for c in range(self.num_classes):
                    if class_counts[c] > 0:
                        fisher_W = fisher_W_acc[c] / class_counts[c]
                        fisher_b = fisher_b_acc[c] / class_counts[c] if fc_bias is not None else None
                        print(f"  Class {c}: {class_counts[c]} samples, Fisher norm W={fisher_W.norm().item():.4f}")
                    else:
                        print(f"  Warning: No samples for class {c}, using zero Fisher")
                        fisher_W = torch.zeros_like(fc_weight)
                        fisher_b = torch.zeros_like(fc_bias) if fc_bias is not None else None

                    fisher_W_list.append(fisher_W)
                    if fisher_b is not None:
                        fisher_b_list.append(fisher_b)

                # Stack into tensors [num_classes, num_classes, feature_dim]
                self.fisher_W_tensor = torch.stack(fisher_W_list, dim=0)
                self.fisher_b_tensor = torch.stack(fisher_b_list, dim=0) if fc_bias is not None else None

                print(f"✓ Computed Fisher matrices for {self.num_classes} classes")

            # Save Fisher matrices
            self._save_fisher_matrices(fisher_cache_path)

    def postprocess(self, net: nn.Module, data: Any):
        """Compute OOD scores for a batch using vmap for parallelization.

        Args:
            net: Neural network
            data: Input batch [B, C, H, W]

        Returns:
            scores: OOD scores [B]
        """
        device = data.device
        batch_size = data.shape[0]

        # Extract features
        net.eval()
        with torch.no_grad():
            _, features = net(data, return_feature=True)

        # Get FC parameters
        fc_weight, fc_bias = self._get_fc_params(net)

        # Replicate FC parameters for each sample in batch
        fc_weights = fc_weight.unsqueeze(0).expand(batch_size, *fc_weight.shape)
        if fc_bias is not None:
            fc_biases = fc_bias.unsqueeze(0).expand(batch_size, *fc_bias.shape)
        else:
            fc_biases = None

        # Compute global prototype
        global_proto = self.prototypes.mean(dim=0) if self.prototypes is not None else None

        # Use vmap to parallelize over batch
        if fc_biases is not None:
            vmapped_fn = vmap(
                lambda w, b, f, proto, g_proto: self._single_sample_unlearn(w, b, f, proto, g_proto, device),
                in_dims=(0, 0, 0, None, None),  # fc_weight, fc_bias, feature, prototypes (broadcast), global_proto (broadcast)
                out_dims=0  # score
            )
            scores = vmapped_fn(fc_weights, fc_biases, features, self.prototypes, global_proto)
        else:
            vmapped_fn = vmap(
                lambda w, f, proto, g_proto: self._single_sample_unlearn(w, None, f, proto, g_proto, device),
                in_dims=(0, 0, None, None),  # fc_weight, feature, prototypes (broadcast), global_proto (broadcast)
                out_dims=0  # score
            )
            scores = vmapped_fn(fc_weights, features, self.prototypes, global_proto)

        return scores

    def inference(self, net: nn.Module, data_loader, progress: bool = True):
        """Run inference on a data loader.

        Args:
            net: Neural network
            data_loader: Data loader
            progress: Show progress bar

        Returns:
            pred_list: Predictions [N]
            conf_list: OOD scores [N]
            label_list: Ground truth labels [N]
        """
        from tqdm import tqdm
        import openood.utils.comm as comm

        pred_list, conf_list, label_list = [], [], []

        for batch in tqdm(data_loader,
                         disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()

            # Get predictions from network
            with torch.no_grad():
                logits, _ = net(data, return_feature=True)
                pred = logits.argmax(dim=1)

            # Compute OOD scores using postprocessor
            scores = self.postprocess(net, data)

            pred_list.append(pred.cpu())
            conf_list.append(scores.cpu())
            label_list.append(label.cpu())

        # Convert to numpy arrays
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list
