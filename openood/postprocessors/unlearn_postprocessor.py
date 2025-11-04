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
        self.prototype_aggregation = str(pc_cfg.get("prototype_aggregation", "median"))  # "mean" or "median"

        # Will be set in setup()
        self.prototypes = None

    @staticmethod
    def _energy(logits: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(logits, dim=-1)

    def _pseudo_target(self, logit_row: torch.Tensor) -> torch.Tensor:
        """Compute soft pseudo-label target with temperature."""
        return F.softmax(logit_row / self.temp, dim=-1)

    def _get_fc_params(self, net):
        """Extract final FC layer parameters."""
        fc = net.fc if hasattr(net, 'fc') else net.module.fc if hasattr(net, 'module') else None
        if fc is None:
            raise ValueError("Cannot find FC layer")
        return fc.weight.data.clone(), fc.bias.data.clone()

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

        # Compute participation ratio: PR = (trace(C))^2 / trace(C^2)
        trace_C = torch.trace(coupling_matrix)
        trace_C2 = torch.trace(torch.matmul(coupling_matrix, coupling_matrix))
        pr = (trace_C ** 2) / (trace_C2 + 1e-10)

        # Return negated score (lower PR = ID → higher score)
        return -pr

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

                # Gradient ascent update
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

                # Gradient ascent update
                W = W + self.eta * grad_W

        # Compute OOD score
        if self.score_type == "prototype_coupling":
            score = self._compute_prototype_coupling_score(
                W_base, b_base, W, b, prototype_tensor, logits_init, device
            )
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
                # Aggregate based on config: mean or median
                if self.prototype_aggregation == "mean":
                    prototype = class_features.mean(dim=0)
                else:  # median (default)
                    prototype = torch.median(class_features, dim=0).values
                print(f"  Class {c}: {mask.sum().item()} samples")
            else:
                # Handle empty classes
                print(f"  Warning: No samples found for class {c}")
                prototype = torch.zeros(all_features.shape[1], device=all_features.device)
            prototypes.append(prototype)

        self.prototypes = torch.stack(prototypes)
        print(f"✓ Computed {self.num_classes} class prototypes using {self.prototype_aggregation}")

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
