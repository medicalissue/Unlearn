from typing import Any
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

        # Will be set in setup()
        self.prototypes = None

    @staticmethod
    def _energy(logits: torch.Tensor) -> torch.Tensor:
        return -torch.logsumexp(logits, dim=-1)

    def _pseudo_target(self, logit_row: torch.Tensor) -> torch.Tensor:
        """Compute soft pseudo-label target with temperature."""
        return F.softmax(logit_row / self.temp, dim=-1)

    def _get_fc_params(self, net):
        """Extract final FC layer parameters."""
        fc = net.fc if hasattr(net, 'fc') else net.module.fc if hasattr(net, 'module') else None
        if fc is None:
            raise ValueError("Cannot find FC layer")
        return fc.weight.data.clone(), fc.bias.data.clone()

    def _compute_participation_ratio(self, eigenvalues: torch.Tensor) -> torch.Tensor:
        """Compute participation ratio: (Σλᵢ)² / Σλᵢ²

        ID samples: LOW PR (coupled to few prototypes) → HIGH score (negated)
        OOD samples: HIGH PR (coupled to many prototypes) → LOW score (negated)
        """
        trace_C = eigenvalues.sum(dim=-1)
        trace_C2 = (eigenvalues ** 2).sum(dim=-1)
        pr = (trace_C ** 2) / (trace_C2 + 1e-10)
        return -pr

    def _compute_prototype_coupling_score(self, W_base, b_base, W_current, b_current,
                                         prototype_tensor, logits):
        """Compute prototype coupling score using gradient eigenvalue analysis."""
        batch_size = logits.shape[0]

        # Select prototypes
        if self.use_all_prototypes:
            # Use all prototypes for each sample: [num_classes, feat_dim]
            # Expand to [batch_size, num_classes, feat_dim]
            selected_prototypes = prototype_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            _, top_indices = torch.topk(logits, k=min(self.top_k, self.num_classes), dim=-1)
            selected_prototypes = prototype_tensor[top_indices]  # [batch_size, top_k, feat_dim]

        # Compute gradients for each prototype
        dW = W_current - W_base  # [batch_size, num_classes, feat_dim]

        # Vectorized computation: dW @ prototypes^T
        # dW: [batch_size, num_classes, feat_dim]
        # selected_prototypes: [batch_size, K, feat_dim]
        # Result: [batch_size, num_classes, K]
        grad_logits = torch.bmm(dW, selected_prototypes.transpose(1, 2))
        # Compute norm across the num_classes dimension
        grad_norms = grad_logits.norm(dim=1)  # [batch_size, K]

        # Eigenvalues approximation
        eigenvalues = grad_norms ** 2  # [batch_size, K]

        # Compute metric based on eigenvalue_mode
        if self.eigenvalue_mode == "participation_ratio":
            return self._compute_participation_ratio(eigenvalues)
        else:
            # Default to participation_ratio
            return self._compute_participation_ratio(eigenvalues)

    def _single_sample_unlearn(self, fc_weight, fc_bias, feature, prototype_tensor,
                              global_proto, device):
        """Unlearn a single sample using gradient ascent."""
        batch_size = feature.shape[0]

        # Initialize parameters
        W = fc_weight.unsqueeze(0).expand(batch_size, -1, -1).clone()
        b = fc_bias.unsqueeze(0).expand(batch_size, -1).clone()
        W_base = W.clone()
        b_base = b.clone()

        # Initial logits
        logits_init = (W @ feature.unsqueeze(-1)).squeeze(-1) + b

        # Compute pseudo targets once
        targets = self._pseudo_target(logits_init)

        # Unlearning loop
        for step in range(self.num_steps):
            # Current logits
            logits = (W @ feature.unsqueeze(-1)).squeeze(-1) + b

            # Compute gradients for ascent mode
            probs = F.softmax(logits, dim=-1)
            grad_logits = probs - targets

            # Gradient ascent: move away from target
            grad_W = torch.bmm(grad_logits.unsqueeze(-1), feature.unsqueeze(1))
            grad_b = grad_logits

            # Update
            W = W + self.eta * grad_W
            b = b + self.eta * grad_b

        # Final logits
        logits_final = (W @ feature.unsqueeze(-1)).squeeze(-1) + b

        # Compute OOD score
        if self.score_type == "prototype_coupling":
            scores = self._compute_prototype_coupling_score(
                W_base, b_base, W, b, prototype_tensor, logits_init
            )
        else:
            # Default: use energy change
            energy_init = self._energy(logits_init)
            energy_final = self._energy(logits_final)
            scores = energy_final - energy_init

        return scores

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """Compute class prototypes from ID training data."""
        import tqdm
        net.eval()

        # Use train data if available, otherwise fall back to val data
        if 'train' in id_loader_dict:
            data_loader = id_loader_dict['train']
        elif 'val' in id_loader_dict:
            print("Warning: Training data not available, using validation data for prototype computation")
            data_loader = id_loader_dict['val']
        else:
            raise ValueError("Neither training nor validation data available for setup")

        # Accumulate features per class - use lists for efficiency
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

        # Compute prototypes by class
        prototypes = []
        for c in range(self.num_classes):
            mask = (all_labels == c)
            if mask.sum() > 0:
                class_features = all_features[mask]
                prototype = class_features.mean(dim=0)
            else:
                # Handle empty classes
                prototype = torch.zeros(all_features.shape[1], device=all_features.device)
            prototypes.append(prototype)

        self.prototypes = torch.stack(prototypes)
        print(f"Computed {self.num_classes} class prototypes")

    def postprocess(self, net: nn.Module, data: Any):
        """Compute OOD scores for a batch."""
        net.eval()

        with torch.no_grad():
            logits, features = net(data, return_feature=True)

        # Get FC parameters
        fc_weight, fc_bias = self._get_fc_params(net)

        # Compute global prototype (mean of all prototypes)
        global_proto = self.prototypes.mean(dim=0)

        # Compute OOD scores
        scores = self._single_sample_unlearn(
            fc_weight, fc_bias, features, self.prototypes, global_proto, data.device
        )

        return scores

    def inference(self, net: nn.Module, data_loader, progress: bool = True):
        """Run inference on a data loader."""
        import tqdm
        pred_list, conf_list, label_list = [], [], []

        iterator = tqdm.tqdm(data_loader, desc="Inference", disable=not progress)
        for batch in iterator:
            data = batch['data'].cuda()
            labels = batch['label']

            with torch.no_grad():
                logits, _ = net(data, return_feature=True)
                conf = F.softmax(logits, dim=1).max(dim=1)[0]
                pred = logits.argmax(dim=1)

            # Compute OOD scores
            scores = self.postprocess(net, data)

            pred_list.append(pred.cpu())
            conf_list.append(scores.cpu())
            label_list.append(labels)

        # Aggregate
        pred = torch.cat(pred_list)
        conf = torch.cat(conf_list)
        label = torch.cat(label_list)

        return pred, conf, label
