from typing import Any
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch 1.13.1 사용 시 functorch import
try:
    from functorch import make_functional_with_buffers, vmap, grad
except ImportError:
    # PyTorch 2.0+ 사용 시
    from torch.func import functional_call as make_functional_with_buffers
    from torch.func import vmap, grad

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict

class UnlearnPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.num_classes = num_classes_dict[self.config.dataset.name]

        # 하이퍼파라미터
        self.eta: float = float(self.args.get("eta", 1e-3))
        self.num_steps: int = int(self.args.get("num_steps", 1))
        self.temp: float = float(self.args.get("temp", 1.0))
        self.use_gradnorm: bool = bool(self.args.get("use_gradnorm", True))
        self.score_type: str = str(self.args.get("score_type", "combo")).lower()
        self.unlearn_mode: str = str(self.args.get("unlearn_mode", "ascent")).lower()  # "ascent" or "fisher"
        self.fisher_damping: float = float(self.args.get("fisher_damping", 1e-8))  # Fisher 안정화를 위한 값
        self.fisher_normalize: bool = bool(self.args.get("fisher_normalize", True))  # Fisher 정규화 여부
        self.fisher_eps: float = float(self.args.get("fisher_eps", 1e-6))  # Fisher 정규화 시 eps
        self.recompute_target: bool = bool(self.args.get("recompute_target", False))  # 매 스텝마다 target 재계산 여부

        # 가중치
        wcfg = self.args.get("weights", {}) or {}
        self.w_dE = float(wcfg.get("denergy", 1.0))
        self.w_G  = float(wcfg.get("g",       0.5))
        self.w_ratio = float(wcfg.get("grad_ratio", 0.5))
        self.w_feature = float(wcfg.get("feature", 1.0))  # Feature gradient weight
        self.w_prototype_coupling = float(wcfg.get("prototype_coupling", 1.0))  # Prototype coupling weight
        self.w_gradient_alignment = float(wcfg.get("gradient_alignment", 1.0))  # Gradient alignment weight
        self.w_confidence_entropy = float(wcfg.get("confidence_entropy", 1.0))  # Confidence-entropy weight

        # Feature-aware gradient options
        self.use_feature_grad = bool(self.args.get("use_feature_grad", True))  # Enable feature gradient computation

        # Phase 1: Adaptive learning rate
        self.eta_mode: str = str(self.args.get("eta_mode", "fixed")).lower()  # "fixed", "confidence", "entropy", "hybrid"
        self.eta_power: float = float(self.args.get("eta_power", 0.5))  # Power for confidence scaling (sqrt by default)

        # Phase 1: Gradient normalization
        self.grad_norm_mode: str = str(self.args.get("grad_norm_mode", "none")).lower()  # "none", "l2", "clip", "adaptive_clip"
        self.grad_clip_value: float = float(self.args.get("grad_clip_value", 1.0))  # Clip threshold

        # Phase 2: EMA
        self.ema_momentum: float = float(self.args.get("ema_momentum", 0.0))  # 0=disabled, 0.9=strong smoothing
        self.ema_weight = None  # Will be initialized in _single_sample_unlearn
        self.ema_bias = None

        # Phase 2: Prototype guidance
        self.use_prototype_guidance: bool = bool(self.args.get("use_prototype_guidance", False))
        self.proto_guidance_strength: float = float(self.args.get("proto_guidance_strength", 1.0))

        # Phase 2: Temperature annealing
        self.temp_anneal: bool = bool(self.args.get("temp_anneal", False))
        self.temp_anneal_rate: float = float(self.args.get("temp_anneal_rate", 0.5))

        # Phase 3: Multi-target ensemble
        self.num_targets: int = int(self.args.get("num_targets", 1))  # Number of top-K targets

        # Phase 3: L2 regularization
        self.l2_penalty: float = float(self.args.get("l2_penalty", 0.0))

        # Feature-aware advanced settings
        fa_cfg = self.args.get("feature_aware_config", {}) or {}
        self.fa_mode: str = str(fa_cfg.get("mode", "baseline")).lower()
        self.fa_use_adaptive_norm: bool = bool(fa_cfg.get("use_adaptive_norm", False))

        # Component weights
        comp_weights = fa_cfg.get("component_weights", {}) or {}
        self.fa_w_feature_norm: float = float(comp_weights.get("feature_norm", 1.0))
        self.fa_w_distance: float = float(comp_weights.get("distance", 1.0))
        self.fa_w_weight_shift: float = float(comp_weights.get("weight_shift", 1.0))

        # Cosine similarity
        self.fa_use_cosine: bool = bool(fa_cfg.get("use_cosine", False))
        self.fa_angular_weight: float = float(fa_cfg.get("angular_weight", 0.5))

        # Distance metric
        self.fa_distance_metric: str = str(fa_cfg.get("distance_metric", "l2")).lower()
        dist_comb = fa_cfg.get("distance_combination", {}) or {}
        self.fa_dist_l1_weight: float = float(dist_comb.get("l1_weight", 0.5))
        self.fa_dist_l2_weight: float = float(dist_comb.get("l2_weight", 0.5))

        # Weighted L1 settings
        weighted_l1_cfg = fa_cfg.get("weighted_l1_config", {}) or {}
        self.fa_weighted_l1_importance_mode: str = str(weighted_l1_cfg.get("importance_mode", "variance")).lower()
        self.fa_weighted_l1_normalize: bool = bool(weighted_l1_cfg.get("normalize_weights", True))

        # Fractional p-norm settings
        self.fa_fractional_p: float = float(fa_cfg.get("fractional_p", 0.5))

        # Adaptive p-norm settings
        adaptive_p_cfg = fa_cfg.get("adaptive_p_config", {}) or {}
        self.fa_adaptive_p_min: float = float(adaptive_p_cfg.get("min_p", 1.0))
        self.fa_adaptive_p_max: float = float(adaptive_p_cfg.get("max_p", 2.0))
        self.fa_adaptive_p_mode: str = str(adaptive_p_cfg.get("mode", "linear")).lower()

        # L0 norm settings
        self.fa_l0_threshold: float = float(fa_cfg.get("l0_threshold", 1e-6))

        # RBF kernel distance settings
        rbf_cfg = fa_cfg.get("rbf_config", {})
        self.fa_rbf_sigma = rbf_cfg.get("sigma", "auto")  # Can be "auto", "median", or float
        self.fa_rbf_use_gamma: bool = bool(rbf_cfg.get("use_gamma", False))
        self.fa_rbf_gamma: float = float(rbf_cfg.get("gamma", 0.01))
        self.fa_rbf_sigma_value: float = None  # Computed in setup() if "auto" or "median"

        # Log-scaled distance settings
        log_cfg = fa_cfg.get("log_scaled_config", {})
        self.fa_log_base_metric: str = str(log_cfg.get("base_metric", "l1")).lower()
        self.fa_log_mode: str = str(log_cfg.get("log_mode", "log1p")).lower()
        self.fa_log_eps: float = float(log_cfg.get("eps", 1e-8))

        # Element-wise log distance settings
        elog_cfg = fa_cfg.get("elementwise_log_config", {})
        self.fa_elog_alpha: float = float(elog_cfg.get("alpha", 1.0))
        self.fa_elog_base: str = str(elog_cfg.get("base", "natural")).lower()

        # Truncated fractional distance settings
        tfrac_cfg = fa_cfg.get("truncated_fractional_config", {})
        self.fa_tfrac_p: float = float(tfrac_cfg.get("p", 0.1))
        self.fa_tfrac_threshold: float = float(tfrac_cfg.get("threshold", 1e-3))
        self.fa_tfrac_mode: str = str(tfrac_cfg.get("mode", "hard")).lower()

        # Mixed p-norm distance settings
        mixp_cfg = fa_cfg.get("mixed_p_config", {})
        self.fa_mixp_p_large: float = float(mixp_cfg.get("p_large", 0.5))
        self.fa_mixp_p_small: float = float(mixp_cfg.get("p_small", 0.05))
        self.fa_mixp_threshold_mode: str = str(mixp_cfg.get("threshold_mode", "median")).lower()
        self.fa_mixp_percentile: float = float(mixp_cfg.get("percentile", 50))
        self.fa_mixp_absolute_threshold: float = float(mixp_cfg.get("absolute_threshold", 0.1))

        # Nonlinear transformation
        self.fa_nonlinear_mode: str = str(fa_cfg.get("nonlinear_mode", "none")).lower()

        # Confidence scaling
        self.fa_use_confidence_scaling: bool = bool(fa_cfg.get("use_confidence_scaling", False))
        self.fa_confidence_power: float = float(fa_cfg.get("confidence_power", 1.0))

        # Statistics for adaptive normalization (initialized in setup)
        self.fa_stats = {
            "feature_norm_mean": None,
            "feature_norm_std": None,
            "distance_mean": None,
            "distance_std": None,
            "weight_shift_mean": None,
            "weight_shift_std": None,
        }

        # Feature importance weights (for weighted_l1, initialized in setup)
        self.feature_importance = None  # [feature_dim] tensor

        # Cross-Prototype Gradient Coupling settings
        pc_cfg = self.args.get("prototype_coupling_config", {}) or {}
        self.pc_use_all_prototypes: bool = bool(pc_cfg.get("use_all_prototypes", True))
        self.pc_top_k: int = int(pc_cfg.get("top_k", 5))
        self.pc_eigenvalue_mode: str = str(pc_cfg.get("eigenvalue_mode", "concentration")).lower()
        self.pc_gpr_q: float = float(pc_cfg.get("gpr_q", 2.0))  # For generalized_pr mode

        # Magnitude-weighted PR settings
        mwpr_cfg = pc_cfg.get("magnitude_weighted_pr_config", {}) or {}
        self.pc_mwpr_q: float = float(mwpr_cfg.get("q", 2.0))
        self.pc_mwpr_combine_mode: str = str(mwpr_cfg.get("combine_mode", "multiply")).lower()

        # Spectrum stats settings
        spectrum_cfg = pc_cfg.get("spectrum_stats_config", {}) or {}
        spectrum_weights = spectrum_cfg.get("weights", {}) or {}
        self.pc_spectrum_w_total_energy: float = float(spectrum_weights.get("total_energy", 1.0))
        self.pc_spectrum_w_max_eig: float = float(spectrum_weights.get("max_eigenvalue", 0.5))
        self.pc_spectrum_w_gini: float = float(spectrum_weights.get("gini", 0.3))
        self.pc_spectrum_w_variance: float = float(spectrum_weights.get("variance", 0.2))

        # Dual metric settings
        dual_cfg = pc_cfg.get("dual_metric_config", {}) or {}
        self.pc_dual_alpha: float = float(dual_cfg.get("alpha", 0.7))
        self.pc_dual_beta: float = float(dual_cfg.get("beta", 0.3))

        # Gradient Alignment settings
        ga_cfg = self.args.get("gradient_alignment_config", {}) or {}
        self.ga_normalize_grads: bool = bool(ga_cfg.get("normalize_grads", True))
        self.ga_use_absolute: bool = bool(ga_cfg.get("use_absolute", False))

        # Trajectory-based OOD detection settings
        traj_cfg = self.args.get("trajectory_config", {}) or {}
        self.traj_enabled: bool = bool(traj_cfg.get("enabled", False))

        # Trajectory component weights
        traj_weights = traj_cfg.get("weights", {}) or {}
        self.traj_w_energy: float = float(traj_weights.get("energy", 1.0))
        self.traj_w_weight: float = float(traj_weights.get("weight", 1.0))
        self.traj_w_gradient: float = float(traj_weights.get("gradient", 1.0))
        self.traj_w_coupling: float = float(traj_weights.get("coupling", 1.0))

        # Hyperparameter sweep configuration
        self.args_dict = self.config.postprocessor.postprocessor_sweep

    def set_hyperparam(self, hyperparam: list):
        """Set hyperparameters from flat list matching config key order

        The hyperparam list order matches the order of keys in postprocessor_sweep config.
        All parameters in postprocessor_sweep must be flat (no nested structures).

        Args:
            hyperparam: List of hyperparameters in the order they appear in config
        """
        if not hasattr(self, 'args_dict') or not self.args_dict:
            return

        param_idx = 0
        for key in self.args_dict.keys():
            if param_idx >= len(hyperparam):
                break

            value = hyperparam[param_idx]
            param_idx += 1

            # Map each config key to corresponding attribute
            if key == 'unlearn_mode':
                self.unlearn_mode = str(value)
            elif key == 'fisher_normalize':
                self.fisher_normalize = bool(value)
            elif key == 'fisher_damping':
                self.fisher_damping = float(value)
            elif key == 'fisher_eps':
                self.fisher_eps = float(value)
            elif key == 'eta':
                self.eta = float(value)
            elif key == 'num_steps':
                self.num_steps = int(value)
            elif key == 'temp':
                self.temp = float(value)
            elif key == 'recompute_target':
                self.recompute_target = bool(value)
            elif key == 'use_gradnorm':
                self.use_gradnorm = bool(value)
            elif key == 'score_type':
                self.score_type = str(value)
            elif key == 'weight_denergy':
                self.w_dE = float(value)
            elif key == 'weight_g':
                self.w_G = float(value)
            elif key == 'weight_grad_ratio':
                self.w_ratio = float(value)

    def get_hyperparam(self):
        """Get current hyperparameters in config key order

        Returns:
            List of hyperparameters matching the order in postprocessor_sweep config
        """
        if not hasattr(self, 'args_dict') or not self.args_dict:
            return []

        result = []
        for key in self.args_dict.keys():
            if key == 'unlearn_mode':
                result.append(self.unlearn_mode)
            elif key == 'fisher_normalize':
                result.append(self.fisher_normalize)
            elif key == 'fisher_damping':
                result.append(self.fisher_damping)
            elif key == 'fisher_eps':
                result.append(self.fisher_eps)
            elif key == 'eta':
                result.append(self.eta)
            elif key == 'num_steps':
                result.append(self.num_steps)
            elif key == 'temp':
                result.append(self.temp)
            elif key == 'recompute_target':
                result.append(self.recompute_target)
            elif key == 'use_gradnorm':
                result.append(self.use_gradnorm)
            elif key == 'score_type':
                result.append(self.score_type)
            elif key == 'weight_denergy':
                result.append(self.w_dE)
            elif key == 'weight_g':
                result.append(self.w_G)
            elif key == 'weight_grad_ratio':
                result.append(self.w_ratio)

        return result

    # ---------- utils ----------

    @staticmethod
    def _energy(logits: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(logits, dim=-1)

    @torch.no_grad()
    def _pseudo_target(self, logit_row: torch.Tensor) -> torch.Tensor:
        if self.temp is None or self.temp <= 1.0:
            k = logit_row.argmax(dim=-1)
            return F.one_hot(k, num_classes=self.num_classes).float()
        else:
            return F.softmax(logit_row / self.temp, dim=-1)

    def _adaptive_eta(self, logits: torch.Tensor, base_eta: float) -> torch.Tensor:
        """Phase 1.1: Adaptive learning rate based on prediction confidence

        Args:
            logits: Model logits [num_classes]
            base_eta: Base learning rate

        Returns:
            Adaptive learning rate (scalar)
        """
        if self.eta_mode == "fixed":
            return base_eta

        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values  # Max probability

        if self.eta_mode == "confidence":
            # High confidence (ID) → large eta, Low confidence (OOD) → small eta
            eta_adaptive = base_eta * (confidence ** self.eta_power)

        elif self.eta_mode == "entropy":
            # Compute normalized entropy
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            entropy_normalized = entropy / torch.log(torch.tensor(self.num_classes, dtype=torch.float32))
            # Low entropy (ID) → large eta, High entropy (OOD) → small eta
            eta_adaptive = base_eta * (1.0 - entropy_normalized)

        elif self.eta_mode == "hybrid":
            # Combine confidence and entropy
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            entropy_normalized = entropy / torch.log(torch.tensor(self.num_classes, dtype=torch.float32))
            eta_adaptive = base_eta * confidence * (1.0 - 0.5 * entropy_normalized)
        else:
            eta_adaptive = base_eta

        return eta_adaptive

    def _normalize_gradient(self, grad_W: torch.Tensor, grad_b: torch.Tensor):
        """Phase 1.2: Normalize gradient for stable updates

        Args:
            grad_W: FC weight gradient [num_classes, feature_dim]
            grad_b: FC bias gradient [num_classes] or None

        Returns:
            Normalized gradients (grad_W, grad_b)
        """
        if self.grad_norm_mode == "none":
            return grad_W, grad_b

        elif self.grad_norm_mode == "l2":
            # L2 normalization (unit gradient direction)
            grad_W_norm = torch.norm(grad_W, p=2)
            grad_W_normalized = grad_W / (grad_W_norm + 1e-8)
            if grad_b is not None:
                grad_b_norm = torch.norm(grad_b, p=2)
                grad_b_normalized = grad_b / (grad_b_norm + 1e-8)
            else:
                grad_b_normalized = None
            return grad_W_normalized, grad_b_normalized

        elif self.grad_norm_mode == "clip":
            # Gradient clipping (prevent extreme updates)
            grad_W_clipped = torch.clamp(grad_W, -self.grad_clip_value, self.grad_clip_value)
            if grad_b is not None:
                grad_b_clipped = torch.clamp(grad_b, -self.grad_clip_value, self.grad_clip_value)
            else:
                grad_b_clipped = None
            return grad_W_clipped, grad_b_clipped

        elif self.grad_norm_mode == "adaptive_clip":
            # Adaptive clipping based on gradient percentile
            grad_W_flat = grad_W.abs().flatten()
            clip_value = torch.quantile(grad_W_flat, 0.95)  # Clip top 5% outliers
            grad_W_clipped = torch.clamp(grad_W, -clip_value, clip_value)
            if grad_b is not None:
                grad_b_flat = grad_b.abs().flatten()
                clip_value_b = torch.quantile(grad_b_flat, 0.95)
                grad_b_clipped = torch.clamp(grad_b, -clip_value_b, clip_value_b)
            else:
                grad_b_clipped = None
            return grad_W_clipped, grad_b_clipped

        return grad_W, grad_b

    def _prototype_guided_gradient(self, grad_W: torch.Tensor, feature: torch.Tensor,
                                   pseudo_class: torch.Tensor, prototype_tensor: torch.Tensor):
        """Phase 2.2: Modify gradient direction based on prototype alignment

        Args:
            grad_W: FC weight gradient [num_classes, feature_dim]
            feature: Test sample feature [feature_dim]
            pseudo_class: Predicted class (scalar)
            prototype_tensor: Class prototypes [num_classes, feature_dim] or None

        Returns:
            Prototype-guided gradient
        """
        if not self.use_prototype_guidance or prototype_tensor is None:
            return grad_W

        # Get prototype for pseudo-class
        one_hot = F.one_hot(pseudo_class, num_classes=prototype_tensor.shape[0]).float()
        proto_feature = torch.matmul(one_hot, prototype_tensor)

        # Feature → prototype direction
        direction = proto_feature - feature
        direction_norm = direction / (torch.norm(direction, p=2) + 1e-8)

        # Compute alignment: how much does grad_W align with moving toward prototype?
        alignment = torch.matmul(grad_W, direction_norm)  # [num_classes]

        # Enhance gradient in aligned direction
        # ID samples: high alignment (gradient naturally points toward prototype)
        # OOD samples: low alignment (gradient is random)
        alignment_weight = 1.0 + self.proto_guidance_strength * torch.sigmoid(alignment)
        grad_W_guided = grad_W * alignment_weight.unsqueeze(-1)

        return grad_W_guided

    def _ema_update(self, W_new: torch.Tensor, b_new: torch.Tensor,
                   W_orig: torch.Tensor, b_orig: torch.Tensor):
        """Phase 2.1: Exponential moving average of weight updates

        Args:
            W_new: Updated weight
            b_new: Updated bias
            W_orig: Original weight
            b_orig: Original bias

        Returns:
            EMA-smoothed (W, b)
        """
        if self.ema_momentum == 0.0:
            return W_new, b_new

        # Initialize EMA on first call (per-sample, so always initialize)
        # Note: In vmap context, this is called per sample independently
        ema_weight = W_orig.clone()
        ema_bias = b_orig.clone() if b_orig is not None else None

        # EMA update: W_ema = momentum * W_orig + (1 - momentum) * W_new
        # This interpolates between original and updated weights
        ema_weight = self.ema_momentum * ema_weight + (1 - self.ema_momentum) * W_new
        if b_new is not None and ema_bias is not None:
            ema_bias = self.ema_momentum * ema_bias + (1 - self.ema_momentum) * b_new

        return ema_weight, ema_bias

    def _get_annealed_temp(self, step: int) -> float:
        """Phase 2.3: Get temperature with annealing

        Args:
            step: Current unlearning step

        Returns:
            Annealed temperature
        """
        if not self.temp_anneal or self.num_steps <= 1:
            return self.temp

        # Exponential decay: temp * (rate ** (step / num_steps))
        temp_annealed = self.temp * (self.temp_anneal_rate ** (step / self.num_steps))
        return temp_annealed

    # ============================================================
    # Feature-aware helper methods
    # ============================================================

    def _compute_adaptive_norm(self, value: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        """Z-score normalization for adaptive scaling

        Args:
            value: Value to normalize
            mean: Mean from training statistics
            std: Standard deviation from training statistics

        Returns:
            Normalized value (z-score)
        """
        if mean is None or std is None:
            return value  # Fallback to no normalization

        # Z-score: (x - μ) / σ
        normalized = (value - mean) / (std + 1e-8)
        return normalized

    def _compute_cosine_similarity(self, feature: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between feature and prototype

        Args:
            feature: Feature vector [feature_dim]
            prototype: Prototype vector [feature_dim]

        Returns:
            Angular distance (1 - cosine_similarity), range [0, 2]
        """
        # Normalize vectors
        feature_norm = F.normalize(feature.unsqueeze(0), p=2, dim=-1).squeeze(0)
        proto_norm = F.normalize(prototype.unsqueeze(0), p=2, dim=-1).squeeze(0)

        # Cosine similarity: f·p / (||f|| * ||p||)
        cos_sim = torch.dot(feature_norm, proto_norm)

        # Angular distance: 1 - cos_sim
        # Range: [0, 2] (0=same direction, 1=orthogonal, 2=opposite)
        angular_dist = 1.0 - cos_sim

        return angular_dist

    def _apply_nonlinear(self, value: torch.Tensor, mode: str) -> torch.Tensor:
        """Apply nonlinear transformation for robust scaling

        Args:
            value: Value to transform
            mode: "none", "sqrt", "log"

        Returns:
            Transformed value
        """
        if mode == "sqrt":
            # Square root: reduces large values more than small ones
            # Sign-preserving: sign(x) * sqrt(|x|)
            return torch.sign(value) * torch.sqrt(torch.abs(value) + 1e-8)
        elif mode == "log":
            # Logarithmic: even stronger compression
            # log(1 + |x|) preserves sign
            return torch.sign(value) * torch.log1p(torch.abs(value))
        else:
            # No transformation
            return value

    def _apply_confidence_scaling(self, score: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """Apply confidence-based temperature scaling

        Args:
            score: Base score to scale
            logits: Logits for confidence computation

        Returns:
            Scaled score
        """
        if not self.fa_use_confidence_scaling:
            return score

        # Compute confidence
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values if logits.dim() > 1 else probs.max()

        # Temperature scaling: 1 / (confidence^power)
        # High confidence → low temperature → smaller score
        # Low confidence → high temperature → larger score
        temperature = 1.0 / (torch.pow(confidence, self.fa_confidence_power) + 1e-8)

        # Scale score
        scaled_score = temperature * score

        return scaled_score

    def _compute_distance(self, vec1: torch.Tensor, vec2: torch.Tensor, metric: str = None) -> torch.Tensor:
        """Compute distance between two vectors using specified metric

        Args:
            vec1: First vector
            vec2: Second vector
            metric: Distance metric ("l1", "l2", "linf", "combined")
                   If None, uses self.fa_distance_metric

        Returns:
            Distance value (scalar tensor)
        """
        if metric is None:
            metric = self.fa_distance_metric

        diff = vec1 - vec2

        if metric == "l1":
            # Manhattan distance: Σ|x_i - y_i|
            # Better for high-dim, sparse features, outlier-robust
            return torch.norm(diff, p=1)
        elif metric == "l2":
            # Euclidean distance: √(Σ(x_i - y_i)²)
            # Standard, intuitive, rotation-invariant
            return torch.norm(diff, p=2)
        elif metric == "linf":
            # Chebyshev distance: max|x_i - y_i|
            # Focus on largest dimension difference
            return torch.norm(diff, p=float('inf'))
        elif metric == "combined":
            # Weighted combination of L1 and L2
            # Balances robustness and smoothness
            l1_dist = torch.norm(diff, p=1)
            l2_dist = torch.norm(diff, p=2)
            return self.fa_dist_l1_weight * l1_dist + self.fa_dist_l2_weight * l2_dist
        elif metric == "weighted_l1":
            # Weighted L1: Feature importance-based weighting
            # Requires self.feature_importance to be computed in setup()
            if self.feature_importance is None:
                # Fallback to standard L1 if importance not computed
                return torch.norm(diff, p=1)

            diff_abs = torch.abs(diff)
            weighted_diff = diff_abs * self.feature_importance
            return weighted_diff.sum()

        elif metric == "fractional":
            # Fractional p-norm: L_p with p < 1
            # More sparse-friendly than L1
            p = self.fa_fractional_p
            diff_abs = torch.abs(diff) + 1e-10  # Avoid 0^p issues
            return torch.pow(diff_abs, p).sum() ** (1.0 / p)

        elif metric == "adaptive_p":
            # Adaptive p-norm: confidence-based p selection
            # Requires logits to be passed (for now, fallback to L1)
            # This will be properly handled when logits are available
            # For now, use min_p (L1-like behavior)
            p = self.fa_adaptive_p_min
            diff_abs = torch.abs(diff) + 1e-10
            return torch.pow(diff_abs, p).sum() ** (1.0 / p)

        elif metric == "l0":
            # L0 norm: count of non-zero elements
            # Most sparse-aware metric - counts # of dimensions that differ
            # Uses threshold to determine "non-zero" (for numerical stability)
            # Higher L0 = more dimensions changed = more different from prototype
            diff_abs = torch.abs(diff)
            l0_norm = (diff_abs > self.fa_l0_threshold).sum().float()
            return l0_norm

        elif metric == "rbf":
            # RBF (Radial Basis Function) kernel distance
            # K(x, y) = exp(-||x-y||^2 / (2*sigma^2))
            # Distance: 1 - K(x, y) (convert similarity to distance)
            # Properties: bounded [0, 1], smooth, non-linear transformation
            # Benefits: compresses large distances, emphasizes local differences

            # Compute L2 distance squared
            l2_squared = torch.sum(diff ** 2)

            # Compute sigma or gamma
            if self.fa_rbf_use_gamma:
                # Use gamma directly (sklearn style)
                gamma = self.fa_rbf_gamma
                kernel_value = torch.exp(-gamma * l2_squared)
            else:
                # Use sigma (traditional RBF style)
                if self.fa_rbf_sigma_value is None:
                    # Fallback to default sigma if not computed in setup
                    sigma = 1.0
                else:
                    sigma = self.fa_rbf_sigma_value
                kernel_value = torch.exp(-l2_squared / (2 * sigma ** 2))

            # Convert kernel similarity to distance: d = 1 - K
            # kernel_value close to 1 → similar → distance close to 0
            # kernel_value close to 0 → dissimilar → distance close to 1
            rbf_distance = 1.0 - kernel_value
            return rbf_distance

        elif metric == "log_scaled":
            # Log-scaled distance: applies logarithm to compress large distances
            # log(1 + distance) - compresses outliers while preserving small distances
            # Benefits: robust to outliers, normalizes distance distribution

            # First compute base distance
            if self.fa_log_base_metric == "l1":
                base_dist = torch.norm(diff, p=1)
            elif self.fa_log_base_metric == "l2":
                base_dist = torch.norm(diff, p=2)
            elif self.fa_log_base_metric == "linf":
                base_dist = torch.norm(diff, p=float('inf'))
            else:
                # Default to L1
                base_dist = torch.norm(diff, p=1)

            # Apply logarithmic transformation
            if self.fa_log_mode == "log1p":
                # log(1 + x) - safe for x=0, natural log
                log_dist = torch.log1p(base_dist)
            elif self.fa_log_mode == "log":
                # log(x + eps) - manual epsilon
                log_dist = torch.log(base_dist + self.fa_log_eps)
            elif self.fa_log_mode == "log10":
                # log10(1 + x) - stronger compression
                log_dist = torch.log10(base_dist + 1.0)
            else:
                # Default to log1p
                log_dist = torch.log1p(base_dist)

            return log_dist

        elif metric == "elementwise_log":
            # Element-wise Log Distance: Σ log(1 + α|x_i - y_i|)
            # Sub-linear, bounded growth, continuous
            # Better outlier robustness than fractional p=0.1

            # Apply scaling factor alpha
            diff_abs = torch.abs(diff) * self.fa_elog_alpha

            # Apply log transformation based on base
            if self.fa_elog_base == "natural":
                # ln(1 + α|x-y|)
                elog_values = torch.log1p(diff_abs)  # log(1 + x)
            elif self.fa_elog_base == "10":
                # log10(1 + α|x-y|)
                elog_values = torch.log10(diff_abs + 1.0)
            elif self.fa_elog_base == "2":
                # log2(1 + α|x-y|)
                elog_values = torch.log2(diff_abs + 1.0)
            else:
                # Default: natural log
                elog_values = torch.log1p(diff_abs)

            # Sum across all dimensions
            distance = elog_values.sum()
            return distance

        elif metric == "truncated_fractional":
            # Truncated Fractional: (Σ (|x_i-y_i| if >threshold else 0)^p)^(1/p)
            # Fractional p=0.1 + noise filtering
            # Remove small noise, focus on meaningful differences

            diff_abs = torch.abs(diff)

            # Apply truncation based on mode
            if self.fa_tfrac_mode == "hard":
                # Hard cutoff: values below threshold become 0
                mask = diff_abs > self.fa_tfrac_threshold
                diff_truncated = diff_abs * mask.float()
            elif self.fa_tfrac_mode == "soft":
                # Soft cutoff: exponential decay
                # diff * exp(-threshold / |diff|)
                decay = torch.exp(-self.fa_tfrac_threshold / (diff_abs + 1e-10))
                diff_truncated = diff_abs * decay
            else:
                # Default: hard cutoff
                mask = diff_abs > self.fa_tfrac_threshold
                diff_truncated = diff_abs * mask.float()

            # Apply fractional p-norm
            p = self.fa_tfrac_p
            # Avoid 0^p issues
            diff_truncated = diff_truncated + 1e-10
            distance = torch.pow(diff_truncated, p).sum() ** (1.0 / p)
            return distance

        elif metric == "mixed_p":
            # Mixed P-norm: Large diffs use p_large, small diffs use p_small
            # Adaptive: different treatment for large vs small differences
            # More flexible than single p-norm

            diff_abs = torch.abs(diff)

            # Determine threshold based on mode
            if self.fa_mixp_threshold_mode == "median":
                # Use median of current diff as threshold
                threshold = torch.median(diff_abs)
            elif self.fa_mixp_threshold_mode == "percentile":
                # Use k-th percentile
                k = self.fa_mixp_percentile
                # torch.quantile requires float k in [0, 1]
                threshold = torch.quantile(diff_abs, k / 100.0)
            elif self.fa_mixp_threshold_mode == "absolute":
                # Use fixed absolute value
                threshold = self.fa_mixp_absolute_threshold
            else:
                # Default: median
                threshold = torch.median(diff_abs)

            # Split into large and small differences
            large_mask = diff_abs >= threshold
            small_mask = diff_abs < threshold

            large_diffs = diff_abs * large_mask.float() + 1e-10
            small_diffs = diff_abs * small_mask.float() + 1e-10

            # Apply different p-norms
            p_large = self.fa_mixp_p_large
            p_small = self.fa_mixp_p_small

            # Compute both norms
            large_norm = torch.pow(large_diffs, p_large).sum() ** (1.0 / p_large)
            small_norm = torch.pow(small_diffs, p_small).sum() ** (1.0 / p_small)

            # Combine: sum of both norms
            distance = large_norm + small_norm
            return distance

        else:
            # Fallback to L2
            return torch.norm(diff, p=2)

    def _compute_adaptive_p(self, logits: torch.Tensor) -> float:
        """Compute adaptive p value based on prediction confidence

        Args:
            logits: Model logits [num_classes]

        Returns:
            p value in [min_p, max_p]
        """
        # Compute confidence
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max().item()

        # Map confidence [0, 1] to p [min_p, max_p]
        if self.fa_adaptive_p_mode == "linear":
            # Linear interpolation: high confidence → max_p (L2-like)
            #                      low confidence → min_p (L1-like)
            p = self.fa_adaptive_p_min + confidence * (self.fa_adaptive_p_max - self.fa_adaptive_p_min)
        elif self.fa_adaptive_p_mode == "sigmoid":
            # Sigmoid: smoother transition
            import math
            # Map confidence to [-3, 3] then sigmoid to [0, 1]
            x = (confidence - 0.5) * 6  # Scale to [-3, 3]
            sigmoid = 1.0 / (1.0 + math.exp(-x))
            p = self.fa_adaptive_p_min + sigmoid * (self.fa_adaptive_p_max - self.fa_adaptive_p_min)
        else:
            # Default: linear
            p = self.fa_adaptive_p_min + confidence * (self.fa_adaptive_p_max - self.fa_adaptive_p_min)

        return p

    def _compute_feature_importance(self, net: nn.Module, id_loader_dict, mode: str):
        """Compute per-dimension feature importance for weighted_l1

        Args:
            net: Neural network
            id_loader_dict: ID data loaders
            mode: "variance", "gradient", "fisher", "uniform"

        Returns:
            Tensor of shape [feature_dim] with importance weights
        """
        if not id_loader_dict or 'val' not in id_loader_dict:
            print("  Warning: No ID validation data for feature importance")
            return None

        id_loader = id_loader_dict['val']
        net.eval()

        # Collect features
        all_features = []

        with torch.no_grad():
            for batch in id_loader:
                data = batch['data'].cuda()

                # Extract features
                try:
                    _, features = net(data, return_feature=True)
                except:
                    if hasattr(net, 'forward_features'):
                        features = net.forward_features(data)
                    else:
                        features = net(data)

                all_features.append(features)

        all_features = torch.cat(all_features, dim=0)  # [N, feature_dim]
        feature_dim = all_features.shape[1]

        if mode == "variance":
            # Variance-based: high variance = important
            importance = torch.var(all_features, dim=0)  # [feature_dim]

        elif mode == "gradient":
            # Gradient magnitude-based (requires more computation)
            # For simplicity, use variance as proxy
            importance = torch.var(all_features, dim=0)

        elif mode == "fisher":
            # Fisher information diagonal (requires gradients)
            # For simplicity, use variance as proxy
            importance = torch.var(all_features, dim=0)

        elif mode == "uniform":
            # Uniform: all dimensions equal (equivalent to L1)
            importance = torch.ones(feature_dim, device=all_features.device)

        else:
            # Default: variance
            importance = torch.var(all_features, dim=0)

        # Normalize weights if requested
        if self.fa_weighted_l1_normalize:
            # L1 normalization: sum = feature_dim (preserve scale)
            importance = importance * (feature_dim / (importance.sum() + 1e-10))

        print(f"  Feature importance computed ({mode}): min={importance.min().item():.4f}, "
              f"max={importance.max().item():.4f}, mean={importance.mean().item():.4f}")

        return importance

    def _compute_rbf_sigma(self, net: nn.Module, id_loader_dict, mode: str):
        """Compute RBF kernel bandwidth (sigma) for rbf distance metric

        Args:
            net: Neural network
            id_loader_dict: ID data loaders
            mode: "auto" (sqrt(feature_dim)) or "median" (median pairwise distance)

        Returns:
            float: sigma value
        """
        if not id_loader_dict or 'val' not in id_loader_dict:
            print("  Warning: No ID validation data for RBF sigma computation")
            # Fallback to auto mode with estimated feature_dim
            return 10.0  # Default fallback

        id_loader = id_loader_dict['val']
        net.eval()

        if mode == "auto":
            # Auto mode: sigma = sqrt(feature_dim)
            # First, get feature dimension
            with torch.no_grad():
                for batch in id_loader:
                    data = batch['data'].cuda()
                    try:
                        _, features = net(data, return_feature=True)
                    except:
                        if hasattr(net, 'forward_features'):
                            features = net.forward_features(data)
                        else:
                            features = net(data)
                    feature_dim = features.shape[1]
                    break  # Just need one batch to get dimension

            sigma = torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32)).item()
            print(f"  Auto sigma = sqrt({feature_dim}) = {sigma:.4f}")
            return sigma

        elif mode == "median":
            # Median mode: compute median of pairwise L2 distances
            # Use a subset of data to avoid O(N^2) computation
            all_features = []
            max_samples = 1000  # Limit to 1000 samples for efficiency

            with torch.no_grad():
                for batch in id_loader:
                    data = batch['data'].cuda()
                    try:
                        _, features = net(data, return_feature=True)
                    except:
                        if hasattr(net, 'forward_features'):
                            features = net.forward_features(data)
                        else:
                            features = net(data)
                    all_features.append(features)

                    if len(all_features) * features.shape[0] >= max_samples:
                        break

            all_features = torch.cat(all_features, dim=0)[:max_samples]  # [N, feature_dim]
            N = all_features.shape[0]

            # Compute pairwise L2 distances (use a random subset to save memory)
            if N > 500:
                # Random sample 500 points for pairwise distance computation
                indices = torch.randperm(N)[:500]
                all_features = all_features[indices]
                N = 500

            # Compute pairwise distances: ||x_i - x_j||^2
            # Broadcasting: [N, 1, D] - [1, N, D] -> [N, N, D]
            diffs = all_features.unsqueeze(1) - all_features.unsqueeze(0)  # [N, N, D]
            pairwise_l2 = torch.sqrt(torch.sum(diffs ** 2, dim=2))  # [N, N]

            # Get upper triangular (excluding diagonal)
            triu_indices = torch.triu_indices(N, N, offset=1)
            pairwise_distances = pairwise_l2[triu_indices[0], triu_indices[1]]

            # Median distance
            median_dist = torch.median(pairwise_distances).item()
            sigma = median_dist
            print(f"  Median pairwise distance (sigma) = {sigma:.4f} (from {len(pairwise_distances)} pairs)")
            return sigma

        else:
            # Unknown mode: fallback to auto
            print(f"  Warning: Unknown RBF sigma mode '{mode}', using auto")
            return self._compute_rbf_sigma(net, id_loader_dict, "auto")

    def _compute_coupling_metric(self, W_base, b_base, W_current, b_current,
                                  prototype_tensor, proto_indices, device):
        """Compute coupling matrix eigenvalue metric for given FC parameters.

        This is the core logic extracted from prototype_coupling score type,
        allowing us to compute the metric at any point during unlearning trajectory.

        Args:
            W_base: Base FC weight [num_classes, feature_dim]
            b_base: Base FC bias [num_classes] or None
            W_current: Current FC weight [num_classes, feature_dim]
            b_current: Current FC bias [num_classes] or None
            prototype_tensor: Class prototypes [num_classes, feature_dim]
            proto_indices: Indices of prototypes to use [K]
            device: torch device

        Returns:
            metric: Eigenvalue metric value (scalar tensor)
        """
        num_classes = prototype_tensor.shape[0]

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

        # Compute metric based on eigenvalue mode
        # (Using the same logic as prototype_coupling score type)
        if self.pc_eigenvalue_mode == "participation_ratio":
            trace_C = torch.trace(coupling_matrix)
            trace_C2 = torch.trace(torch.matmul(coupling_matrix, coupling_matrix))
            pr = (trace_C ** 2) / (trace_C2 + 1e-10)
            metric = pr  # Not negated here - we'll handle sign in caller

        elif self.pc_eigenvalue_mode == "spectral_entropy":
            eigenvalues = torch.linalg.eigvalsh(coupling_matrix)
            abs_eigs = torch.abs(eigenvalues)
            eig_sum = abs_eigs.sum() + 1e-10
            eig_probs = abs_eigs / eig_sum
            entropy = -(eig_probs * torch.log(eig_probs + 1e-10)).sum()
            metric = entropy  # Not negated here

        elif self.pc_eigenvalue_mode == "gini_coefficient":
            grad_norms = torch.norm(delta_grads, p=2, dim=1)
            K = grad_norms.shape[0]
            diff_matrix = torch.abs(grad_norms.unsqueeze(0) - grad_norms.unsqueeze(1))
            mean_grad = grad_norms.mean() + 1e-10
            gini = diff_matrix.sum() / (2 * K * K * mean_grad)
            metric = gini

        elif self.pc_eigenvalue_mode == "max_mean_ratio":
            grad_norms = torch.norm(delta_grads, p=2, dim=1)
            max_norm = grad_norms.max()
            mean_norm = grad_norms.mean() + 1e-10
            ratio = max_norm / mean_norm
            metric = ratio

        else:
            # Default: participation ratio
            trace_C = torch.trace(coupling_matrix)
            trace_C2 = torch.matmul(coupling_matrix, coupling_matrix).trace()
            pr = (trace_C ** 2) / (trace_C2 + 1e-10)
            metric = pr

        return metric

    def _get_fc_params(self, net):
        """FC layer의 파라미터만 추출"""
        # 마지막 FC layer 파라미터 추출
        if hasattr(net, 'fc'):
            fc = net.fc
        elif hasattr(net, 'classifier'):
            fc = net.classifier
        elif hasattr(net, 'head'):
            fc = net.head
        else:
            # 마지막 Linear layer 찾기
            fc = None
            for module in net.modules():
                if isinstance(module, nn.Linear):
                    fc = module

        if fc is None:
            raise ValueError("Cannot find FC layer in the model")

        return fc.weight.data.clone(), fc.bias.data.clone() if fc.bias is not None else None

    def _single_sample_unlearn(self, fc_weight, fc_bias, feature, prototype_tensor, global_proto, device):
        """단일 샘플에 대한 unlearning 수행 (vmap으로 병렬화될 함수)

        Args:
            fc_weight: FC layer weight [num_classes, feature_dim]
            fc_bias: FC layer bias [num_classes] or None
            feature: 추출된 feature [feature_dim]
            prototype_tensor: 클래스별 프로토타입 피처 [num_classes, feature_dim] or None
            global_proto: 전체 ID 데이터의 global prototype [feature_dim] or None
            device: 디바이스

        Returns:
            pred: 예측 클래스
            score: OOD 스코어
        """
        # 원본 로짓 계산
        if fc_bias is not None:
            logits_orig = F.linear(feature, fc_weight, fc_bias)
        else:
            logits_orig = F.linear(feature, fc_weight)

        # pseudo target 생성 (초기 target 저장)
        target_init = self._pseudo_target(logits_orig).to(device=device, dtype=logits_orig.dtype)
        target = target_init

        # FC 파라미터 복사 (vmap에서는 requires_grad 사용 불가)
        W = fc_weight
        b = fc_bias

        # Feature space 기하학적 특성 계산
        feature_norm = torch.norm(feature, p=2) if self.use_feature_grad else None

        # Prototype distance 계산 (if available)
        dist_to_prototype_orig = None
        dist_to_global_orig = None
        if self.use_feature_grad and prototype_tensor is not None:
            # 예측된 클래스의 prototype까지의 거리
            pseudo_class = target.argmax(dim=-1)
            one_hot = F.one_hot(pseudo_class, num_classes=prototype_tensor.shape[0]).float()
            proto_feature = torch.matmul(one_hot, prototype_tensor)
            # Use configurable distance metric (L1/L2/Linf/Combined)
            dist_to_prototype_orig = self._compute_distance(feature, proto_feature)

            # Global prototype까지의 거리
            if global_proto is not None:
                dist_to_global_orig = self._compute_distance(feature, global_proto)

        # gradnorm 계산 (원본 모델)
        gradnorm_o = None
        if self.use_gradnorm:
            # 그래디언트 계산을 위한 손실 함수 정의
            if b is not None:
                def loss_fn(params):
                    W_inner, b_inner = params
                    out = F.linear(feature, W_inner, b_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = target.argmax(dim=-1, keepdim=True)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss

                grads = grad(loss_fn)((W, b))
                gradnorm_o = torch.abs(grads[0]).sum() + torch.abs(grads[1]).sum()
            else:
                def loss_fn(W_inner):
                    out = F.linear(feature, W_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = target.argmax(dim=-1, keepdim=True)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss

                grad_W = grad(loss_fn)(W)
                gradnorm_o = torch.abs(grad_W).sum()

        # Fisher 정보 계산 (fisher 모드인 경우)
        fisher_W, fisher_b = None, None
        if self.unlearn_mode == "fisher":
            if b is not None:
                def loss_fn_fisher(params):
                    W_inner, b_inner = params
                    out = F.linear(feature, W_inner, b_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = target.argmax(dim=-1, keepdim=True)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss

                grads = grad(loss_fn_fisher)((W, b))
                fisher_W = grads[0] ** 2
                fisher_b = grads[1] ** 2

                # Adaptive Fisher 정규화
                if self.fisher_normalize:
                    # Mean-based normalization
                    fisher_W_mean = fisher_W.mean()
                    fisher_b_mean = fisher_b.mean()
                    fisher_W = fisher_W / (fisher_W_mean + self.fisher_eps)
                    fisher_b = fisher_b / (fisher_b_mean + self.fisher_eps)
            else:
                def loss_fn_fisher(W_inner):
                    out = F.linear(feature, W_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = target.argmax(dim=-1, keepdim=True)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss

                grad_W = grad(loss_fn_fisher)(W)
                fisher_W = grad_W ** 2

                # Adaptive Fisher 정규화
                if self.fisher_normalize:
                    fisher_W_mean = fisher_W.mean()
                    fisher_W = fisher_W / (fisher_W_mean + self.fisher_eps)

        # Trajectory 통계 초기화 (for trajectory-based score types)
        # Energy trajectory
        E_prev = self._energy(logits_orig)
        dE_prev = torch.tensor(0.0, device=device)
        sum_abs_ddE = torch.tensor(0.0, device=device)
        sum_ddE_sq = torch.tensor(0.0, device=device)

        # Weight update trajectory
        W_prev = fc_weight
        delta_W_prev = None
        sum_delta_norm = torch.tensor(0.0, device=device)
        sum_direction_cos = torch.tensor(0.0, device=device)
        first_delta_norm = None

        # Gradient magnitude trajectory
        g_0 = gradnorm_o if self.use_gradnorm else torch.tensor(0.0, device=device)
        g_final = torch.tensor(0.0, device=device)

        # Coupling evolution trajectory (FULL version - per-step coupling computation)
        # Determine proto indices once (reuse for all steps)
        proto_indices = None
        if prototype_tensor is not None:
            num_classes = prototype_tensor.shape[0]
            if self.pc_use_all_prototypes:
                # Use all prototypes (limit to 10 for efficiency)
                proto_indices = torch.arange(min(num_classes, 10), device=device)
            else:
                # Use top-K predicted classes
                _, top_k_indices = torch.topk(logits_orig, k=min(self.pc_top_k, num_classes), dim=-1)
                proto_indices = top_k_indices[:min(10, top_k_indices.shape[0])]

        # Coupling metric trajectory statistics
        coupling_metric_prev = None
        sum_coupling_metric = torch.tensor(0.0, device=device)
        sum_coupling_metric_sq = torch.tensor(0.0, device=device)
        sum_abs_coupling_diff = torch.tensor(0.0, device=device)
        coupling_step_count = 0

        # Counters
        traj_step_count = 0

        # Unlearning 수행
        target_final = target  # 마지막 target 저장용
        for step in range(max(1, self.num_steps)):
            # Phase 2.3: Temperature annealing
            current_temp = self._get_annealed_temp(step)

            # 매 스텝마다 target 재계산 옵션
            if self.recompute_target and step > 0:
                # 현재 파라미터로 로짓 재계산
                with torch.no_grad():
                    if b is not None:
                        logits_current = F.linear(feature, W, b)
                    else:
                        logits_current = F.linear(feature, W)
                    # Use annealed temperature for pseudo-target
                    if current_temp > 1.0:
                        target = F.softmax(logits_current / current_temp, dim=-1)
                    else:
                        k = logits_current.argmax(dim=-1)
                        target = F.one_hot(k, num_classes=self.num_classes).float()
                    target = target.to(device=device, dtype=logits_current.dtype)
                    target_final = target  # 업데이트된 target 저장

            # Phase 1.1: Adaptive eta (compute current logits for confidence)
            with torch.no_grad():
                if b is not None:
                    logits_for_eta = F.linear(feature, W, b)
                else:
                    logits_for_eta = F.linear(feature, W)
            eta_current = self._adaptive_eta(logits_for_eta, self.eta)

            if b is not None:
                def loss_fn_unlearn(params):
                    W_inner, b_inner = params
                    out = F.linear(feature, W_inner, b_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = target.argmax(dim=-1, keepdim=True)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss

                grads = grad(loss_fn_unlearn)((W, b))

                # Phase 1.2: Gradient normalization
                grads_W_norm, grads_b_norm = self._normalize_gradient(grads[0], grads[1])

                # Phase 2.2: Prototype-guided gradient
                grads_W_guided = self._prototype_guided_gradient(
                    grads_W_norm, feature, target.argmax(dim=-1), prototype_tensor
                )

                # 파라미터 업데이트
                if self.unlearn_mode == "fisher":
                    # Fisher-weighted update
                    W = W + eta_current * grads_W_guided * (fisher_W + self.fisher_damping)
                    b = b + eta_current * grads_b_norm * (fisher_b + self.fisher_damping)
                else:
                    # Gradient ascent
                    W = W + eta_current * grads_W_guided
                    b = b + eta_current * grads_b_norm
            else:
                def loss_fn_unlearn(W_inner):
                    out = F.linear(feature, W_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = target.argmax(dim=-1, keepdim=True)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss

                grad_W = grad(loss_fn_unlearn)(W)

                # Phase 1.2: Gradient normalization
                grad_W_norm, _ = self._normalize_gradient(grad_W, None)

                # Phase 2.2: Prototype-guided gradient
                grad_W_guided = self._prototype_guided_gradient(
                    grad_W_norm, feature, target.argmax(dim=-1), prototype_tensor
                )

                # 파라미터 업데이트
                if self.unlearn_mode == "fisher":
                    # Fisher-weighted update
                    W = W + eta_current * grad_W_guided / (fisher_W + self.fisher_damping)
                else:
                    # Gradient ascent
                    W = W + eta_current * grad_W_guided

            # Trajectory 통계 누적 (end of each unlearning step)
            traj_step_count = traj_step_count + 1

            # [1] Energy trajectory
            with torch.no_grad():
                if b is not None:
                    logits_curr = F.linear(feature, W, b)
                else:
                    logits_curr = F.linear(feature, W)
                E_curr = self._energy(logits_curr)
                dE_curr = E_curr - E_prev
                ddE = dE_curr - dE_prev
                sum_abs_ddE = sum_abs_ddE + torch.abs(ddE)
                sum_ddE_sq = sum_ddE_sq + ddE ** 2
                E_prev = E_curr
                dE_prev = dE_curr

            # [2] Weight update trajectory
            delta_W_curr = W - W_prev
            delta_norm_curr = torch.norm(delta_W_curr, p='fro')
            sum_delta_norm = sum_delta_norm + delta_norm_curr

            # Track first delta norm
            if step == 0:
                first_delta_norm = delta_norm_curr

            # Direction consistency (skip first step)
            if delta_W_prev is not None:
                # Cosine similarity between consecutive weight updates
                cos_sim = torch.sum(delta_W_curr * delta_W_prev) / (
                    torch.norm(delta_W_curr) * torch.norm(delta_W_prev) + 1e-10
                )
                sum_direction_cos = sum_direction_cos + cos_sim

            delta_W_prev = delta_W_curr
            W_prev = W

            # [3] Gradient magnitude (already stored in grads)
            if b is not None:
                g_final = torch.abs(grads[0]).sum() + torch.abs(grads[1]).sum()
            else:
                g_final = torch.abs(grad_W).sum()

            # [4] Coupling evolution (FULL version - per-step coupling metric computation)
            # Only compute if needed for coupling-related score types
            if (self.score_type in ["coupling_evolution", "trajectory_combo"] and
                prototype_tensor is not None and proto_indices is not None):
                # Compute coupling metric at current step
                coupling_metric_curr = self._compute_coupling_metric(
                    fc_weight, fc_bias,  # Base (original)
                    W, b,  # Current (after this step)
                    prototype_tensor,
                    proto_indices,
                    device
                )

                # Update coupling statistics
                sum_coupling_metric = sum_coupling_metric + coupling_metric_curr
                sum_coupling_metric_sq = sum_coupling_metric_sq + coupling_metric_curr ** 2

                # Track step-to-step changes
                if coupling_metric_prev is not None:
                    sum_abs_coupling_diff = sum_abs_coupling_diff + torch.abs(coupling_metric_curr - coupling_metric_prev)

                coupling_metric_prev = coupling_metric_curr
                coupling_step_count = coupling_step_count + 1

        # Phase 2.1: EMA smoothing of final weights
        W, b = self._ema_update(W, b, fc_weight, fc_bias)

        # 업데이트된 FC로 로짓 계산
        with torch.no_grad():
            if b is not None:
                logits_after = F.linear(feature, W, b)
            else:
                logits_after = F.linear(feature, W)

        # FC weight change 계산 (ΔW = W_after - W_orig)
        delta_W = W - fc_weight  # [num_classes, feature_dim]
        weight_shift_norm = torch.norm(delta_W, p='fro') if self.use_feature_grad else None  # Frobenius norm

        # Weight-feature alignment 계산
        # ΔW와 feature의 내적: feature 방향으로 얼마나 weight가 변했는가
        if self.use_feature_grad:
            # delta_W: [num_classes, feature_dim], feature: [feature_dim]
            # Compute: ||ΔW · f||_2 (how much logit changes due to weight shift)
            weight_feature_product = torch.matmul(delta_W, feature)  # [num_classes]
            weight_feature_alignment = torch.norm(weight_feature_product, p=2)
        else:
            weight_feature_alignment = None

        # 예측
        pred = torch.argmax(logits_orig, dim=-1, keepdim=True)

        # 스코어 계산 (NaN 방지를 위해 logit 정규화)
        # Max를 빼서 상대적 크기는 유지하면서 수치적 안정성 확보
        logits_orig_max = logits_orig.max(dim=-1, keepdim=True).values
        logits_after_max = logits_after.max(dim=-1, keepdim=True).values

        logits_orig_stable = logits_orig - logits_orig_max
        logits_after_stable = logits_after - logits_after_max

        dE = self._energy(logits_after_stable) - self._energy(logits_orig_stable)

        # 스코어 타입에 따라 계산
        if self.score_type in {"delta_energy", "denergy"}:
            # CORRECTED: Gradient ascent DECREASES energy (makes predictions more confident)
            # ID samples: large energy decrease (dE << 0)
            # OOD samples: small energy decrease (dE < 0, closer to 0)
            # Since framework expects ID > OOD, we NEGATE
            score = -dE
        elif self.score_type == "feature_aware":
            # Feature-aware score with advanced modes
            # Supports: baseline, adaptive_norm, weighted, angular, nonlinear, full
            if weight_shift_norm is not None and dist_to_prototype_orig is not None:
                # ===== COMPONENT 1: Feature norm =====
                if self.fa_use_adaptive_norm or self.fa_mode != "baseline":
                    # Adaptive normalization (z-score)
                    fn_normalized = self._compute_adaptive_norm(
                        feature_norm,
                        self.fa_stats["feature_norm_mean"],
                        self.fa_stats["feature_norm_std"]
                    )
                else:
                    # Baseline: hardcoded scaling
                    fn_normalized = feature_norm / 100.0

                # ===== COMPONENT 2: Distance to prototype =====
                distance_metric = dist_to_prototype_orig  # L2 distance (default)

                # Cosine similarity (angular mode)
                # NOTE: This entire block must be unconditional for vmap compatibility
                if self.fa_use_cosine or self.fa_mode in ["angular", "full"]:
                    # VMAP-COMPATIBLE: No data-dependent conditionals
                    # Assume prototype_tensor is always provided when angular mode is enabled
                    pseudo_class = target.argmax(dim=-1) if target.dim() > 0 else target.argmax()

                    # Safe indexing: clamp to valid range (FULLY UNCONDITIONAL for vmap)
                    # Avoid: pseudo_class < prototype_tensor.shape[0] (data-dependent!)
                    # Assumption: prototype_tensor is not None (checked before vmap call)
                    pseudo_class_safe = torch.clamp(pseudo_class, 0, prototype_tensor.shape[0] - 1)

                    # VMAP-COMPATIBLE indexing: Use one-hot encoding instead of direct indexing
                    # Direct indexing (prototype_tensor[pseudo_class_safe]) calls .item() internally
                    # One-hot approach: safe for vmap
                    one_hot = F.one_hot(pseudo_class_safe, num_classes=prototype_tensor.shape[0]).float()
                    proto_feature = torch.matmul(one_hot, prototype_tensor)  # [num_classes] × [num_classes, D] = [D]

                    angular_dist = self._compute_cosine_similarity(feature, proto_feature)

                    # Combine L2 and angular distance
                    # L2: magnitude difference, Angular: direction difference
                    distance_metric = (
                        (1.0 - self.fa_angular_weight) * dist_to_prototype_orig +
                        self.fa_angular_weight * angular_dist
                    )

                # Adaptive normalization for distance
                if self.fa_use_adaptive_norm or self.fa_mode != "baseline":
                    dist_normalized = self._compute_adaptive_norm(
                        distance_metric,
                        self.fa_stats["distance_mean"],
                        self.fa_stats["distance_std"]
                    )
                else:
                    dist_normalized = distance_metric

                # ===== COMPONENT 3: Weight shift =====
                if self.fa_use_adaptive_norm or self.fa_mode != "baseline":
                    ws_normalized = self._compute_adaptive_norm(
                        weight_shift_norm,
                        self.fa_stats["weight_shift_mean"],
                        self.fa_stats["weight_shift_std"]
                    )
                else:
                    ws_normalized = weight_shift_norm

                # ===== COMBINE COMPONENTS =====
                # Mode: baseline, adaptive_norm, weighted, angular, nonlinear, full
                if self.fa_mode in ["weighted", "full"]:
                    # Weighted combination (additive)
                    raw_score = (
                        self.fa_w_feature_norm * fn_normalized +
                        self.fa_w_distance * dist_normalized +
                        self.fa_w_weight_shift * ws_normalized
                    )
                else:
                    # Baseline or adaptive_norm: multiplicative
                    raw_score = fn_normalized * dist_normalized * ws_normalized

                # ===== NONLINEAR TRANSFORMATION =====
                if self.fa_mode in ["nonlinear", "full"]:
                    raw_score = self._apply_nonlinear(raw_score, self.fa_nonlinear_mode)

                # ===== FINAL SCORE =====
                # ID: small distance, small weight shift → low raw score → HIGH after negation
                # OOD: large distance, large weight shift → high raw score → LOW after negation
                score = -raw_score  # NEGATE: Framework expects ID > OOD

                # ===== CONFIDENCE SCALING =====
                if self.fa_mode == "full":
                    score = self._apply_confidence_scaling(score, logits_orig)
            else:
                # Fallback: feature_grad disabled or missing data
                score = -dE
        elif self.score_type == "feature_fc_alignment":
            # Weight-feature alignment: how much logit changes due to FC shift in feature direction
            # CORRECTED: Confident ID samples have LARGER FC weight shifts (counter-intuitive!)
            # ID: large ||ΔW·f|| (gradient ascent strongly affects confident predictions)
            # OOD: small ||ΔW·f|| (gradient ascent weakly affects uncertain predictions)
            # Therefore we NEGATE to get ID > OOD as expected by framework
            if weight_feature_alignment is not None:
                score = -weight_feature_alignment  # NEGATE: Framework expects ID > OOD
            else:
                score = dE  # Fallback
        elif self.score_type == "geometry_combo":
            # Combined geometric score
            # CORRECTED: All geometric terms favor OOD (high values), so we negate
            if (weight_shift_norm is not None and
                weight_feature_alignment is not None and
                dist_to_prototype_orig is not None):
                # Multi-scale geometric consistency
                # CORRECTED: dE is negative for both ID and OOD, but ID has more negative dE
                # So -dE gives ID > OOD, which is what we want before final negation
                raw_score = (
                    self.w_dE * (-dE) +                                  # Energy change (CORRECTED: -dE so ID > OOD)
                    self.w_feature * weight_shift_norm * feature_norm +  # Weight shift × feature (OOD > ID paradoxically)
                    weight_feature_alignment * dist_to_prototype_orig    # Alignment × distance (OOD > ID)
                )
                score = -raw_score  # NEGATE: Framework expects ID > OOD
            else:
                score = -dE  # Fallback (CORRECTED)
        elif self.score_type == "multiscale_prototype":
            # Multi-scale prototype distance
            # Uses both class-level and global-level prototypes
            # CORRECTED: Distance metrics are higher for OOD, so we negate
            if dist_to_prototype_orig is not None and dist_to_global_orig is not None:
                # OOD: far from both class prototype and global prototype → high distance
                # ID: close to class prototype, also reasonably close to global → low distance
                raw_score = (
                    dist_to_prototype_orig * 2.0 +     # Class-level distance (higher weight)
                    dist_to_global_orig +              # Global-level distance
                    weight_shift_norm                  # FC weight change (also higher for confident ID!)
                )
                score = -raw_score  # NEGATE: Framework expects ID > OOD
            elif dist_to_prototype_orig is not None:
                # Fallback: only class prototype available
                raw_score = dist_to_prototype_orig * weight_shift_norm if weight_shift_norm is not None else dist_to_prototype_orig
                score = -raw_score  # NEGATE
            else:
                score = dE  # Fallback
        elif self.score_type == "prototype_logit_shift":
            # 프로토타입 기반 로짓 변화 메트릭
            if prototype_tensor is not None:
                # Pseudo label 클래스 선택
                pseudo_class = target.argmax(dim=-1)  # scalar index

                # 해당 클래스의 프로토타입 피처 선택 (vmap 호환 방식)
                # prototype_tensor: [num_classes, feature_dim]
                # pseudo_class: scalar
                # one-hot encoding을 사용하여 선택
                one_hot = F.one_hot(pseudo_class, num_classes=prototype_tensor.shape[0]).float()  # [num_classes]
                proto_feature = torch.matmul(one_hot, prototype_tensor)  # [feature_dim]

                # 원본 FC로 프로토타입 로짓 계산
                if b is not None:
                    proto_logit_orig = F.linear(proto_feature, fc_weight, fc_bias)
                else:
                    proto_logit_orig = F.linear(proto_feature, fc_weight)

                # 업데이트된 FC로 프로토타입 로짓 계산
                if b is not None:
                    proto_logit_after = F.linear(proto_feature, W, b)
                else:
                    proto_logit_after = F.linear(proto_feature, W)

                # Energy 변화 계산 (안정화를 위해 max 빼기)
                proto_logit_orig_max = proto_logit_orig.max(dim=-1, keepdim=True).values
                proto_logit_after_max = proto_logit_after.max(dim=-1, keepdim=True).values

                proto_logit_orig_stable = proto_logit_orig - proto_logit_orig_max
                proto_logit_after_stable = proto_logit_after - proto_logit_after_max

                proto_dE = self._energy(proto_logit_after_stable) - self._energy(proto_logit_orig_stable)
                # CORRECTED: Gradient ascent affects ID sample's prototype MORE (it's nearby)
                # ID: prototype nearby → large proto_dE (unlearned together)
                # OOD: prototype far → small proto_dE (unaffected)
                # Original assumption was REVERSED!
                score = -proto_dE  # NEGATE: Framework expects ID > OOD
            else:
                # Fallback: prototype이 없으면 샘플 자체의 delta_energy 사용
                score = dE
        elif self.score_type == "prototype_grad":
            # 프로토타입 기반 gradient 메트릭
            if prototype_tensor is not None:
                # Pseudo label 클래스 선택
                pseudo_class = target.argmax(dim=-1)

                # 해당 클래스의 프로토타입 피처 선택 (vmap 호환 방식)
                one_hot = F.one_hot(pseudo_class, num_classes=prototype_tensor.shape[0]).float()
                proto_feature = torch.matmul(one_hot, prototype_tensor)

                # 원본 FC의 프로토타입에 대한 gradient norm 계산
                if b is not None:
                    def proto_loss_fn_orig(params):
                        W_inner, b_inner = params
                        out = F.linear(proto_feature, W_inner, b_inner)
                        log_probs = F.log_softmax(out, dim=-1)
                        hard = pseudo_class.unsqueeze(-1)
                        loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                        return loss

                    grads_orig = grad(proto_loss_fn_orig)((fc_weight, fc_bias))
                    proto_gradnorm_orig = torch.abs(grads_orig[0]).sum() + torch.abs(grads_orig[1]).sum()
                else:
                    def proto_loss_fn_orig(W_inner):
                        out = F.linear(proto_feature, W_inner)
                        log_probs = F.log_softmax(out, dim=-1)
                        hard = pseudo_class.unsqueeze(-1)
                        loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                        return loss

                    grad_W_orig = grad(proto_loss_fn_orig)(fc_weight)
                    proto_gradnorm_orig = torch.abs(grad_W_orig).sum()

                # 업데이트된 FC의 프로토타입에 대한 gradient norm 계산
                if b is not None:
                    def proto_loss_fn_after(params):
                        W_inner, b_inner = params
                        out = F.linear(proto_feature, W_inner, b_inner)
                        log_probs = F.log_softmax(out, dim=-1)
                        hard = pseudo_class.unsqueeze(-1)
                        loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                        return loss

                    grads_after = grad(proto_loss_fn_after)((W, b))
                    proto_gradnorm_after = torch.abs(grads_after[0]).sum() + torch.abs(grads_after[1]).sum()
                else:
                    def proto_loss_fn_after(W_inner):
                        out = F.linear(proto_feature, W_inner)
                        log_probs = F.log_softmax(out, dim=-1)
                        hard = pseudo_class.unsqueeze(-1)
                        loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                        return loss

                    grad_W_after = grad(proto_loss_fn_after)(W)
                    proto_gradnorm_after = torch.abs(grad_W_after).sum()

                # Gradient norm 변화 계산
                # CORRECTED: Same logic as proto_dE
                # ID sample unlearning affects nearby prototype → larger gradient norm increase
                # OOD sample unlearning doesn't affect distant prototype → smaller gradient increase
                # Original assumption was REVERSED!
                score = -(proto_gradnorm_after - proto_gradnorm_orig)  # NEGATE: Framework expects ID > OOD
            else:
                # Fallback: prototype이 없으면 샘플 자체의 gradnorm 사용
                if gradnorm_o is not None:
                    if b is not None:
                        def loss_fn_after(params):
                            W_inner, b_inner = params
                            out = F.linear(feature, W_inner, b_inner)
                            log_probs = F.log_softmax(out, dim=-1)
                            hard = target.argmax(dim=-1, keepdim=True)
                            loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                            return loss

                        grads_after = grad(loss_fn_after)((W, b))
                        gradnorm_val = torch.abs(grads_after[0]).sum() + torch.abs(grads_after[1]).sum()
                    else:
                        def loss_fn_after(W_inner):
                            out = F.linear(feature, W_inner)
                            log_probs = F.log_softmax(out, dim=-1)
                            hard = target.argmax(dim=-1, keepdim=True)
                            loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                            return loss

                        grad_W_after = grad(loss_fn_after)(W)
                        gradnorm_val = torch.abs(grad_W_after).sum()

                    score = gradnorm_val - gradnorm_o
                else:
                    score = torch.tensor(0.0, device=device)
        elif self.score_type == "prototype_coupling":
            # Cross-Prototype Gradient Coupling (IMPROVED)
            # Uses FULL gradient vectors to build true coupling matrix
            # Measures eigenvalue concentration, participation ratio, or gradient alignment
            # CORRECTED: ID samples have DIFFUSE changes (stable, uniform), OOD have CONCENTRATED changes
            # ID: diffuse changes (high participation ratio, high alignment) → high score
            # OOD: concentrated changes (low participation ratio, low alignment) → low score
            if prototype_tensor is not None:
                num_classes = prototype_tensor.shape[0]
                pseudo_class = target.argmax(dim=-1)

                # Determine which prototypes to use
                if self.pc_use_all_prototypes:
                    # Use all prototypes
                    proto_indices = torch.arange(num_classes, device=device)
                else:
                    # Use top-K predicted classes (based on original logits)
                    _, top_k_indices = torch.topk(logits_orig, k=min(self.pc_top_k, num_classes), dim=-1)
                    proto_indices = top_k_indices

                # Collect FULL gradient vectors for all selected prototypes (before and after)
                grad_vecs_orig = []
                grad_vecs_after = []

                for i in range(proto_indices.shape[0]):
                    # Get prototype index (vmap-safe: use one-hot indexing)
                    idx = proto_indices[i] if self.pc_use_all_prototypes else proto_indices[i]
                    one_hot_i = F.one_hot(idx, num_classes=num_classes).float()
                    proto_feature_i = torch.matmul(one_hot_i, prototype_tensor)

                    # Compute FULL gradient on ORIGINAL FC with this prototype
                    if b is not None:
                        def proto_loss_i_orig(params):
                            W_inner, b_inner = params
                            out = F.linear(proto_feature_i, W_inner, b_inner)
                            log_probs = F.log_softmax(out, dim=-1)
                            hard = idx.unsqueeze(-1)
                            loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                            return loss
                        grads_i_orig = grad(proto_loss_i_orig)((fc_weight, fc_bias))
                        # Flatten to single vector [param_size]
                        grad_vec_i_orig = torch.cat([grads_i_orig[0].flatten(), grads_i_orig[1].flatten()])
                    else:
                        def proto_loss_i_orig(W_inner):
                            out = F.linear(proto_feature_i, W_inner)
                            log_probs = F.log_softmax(out, dim=-1)
                            hard = idx.unsqueeze(-1)
                            loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                            return loss
                        grad_W_i_orig = grad(proto_loss_i_orig)(fc_weight)
                        grad_vec_i_orig = grad_W_i_orig.flatten()

                    # Compute FULL gradient on UPDATED FC with this prototype
                    if b is not None:
                        def proto_loss_i_after(params):
                            W_inner, b_inner = params
                            out = F.linear(proto_feature_i, W_inner, b_inner)
                            log_probs = F.log_softmax(out, dim=-1)
                            hard = idx.unsqueeze(-1)
                            loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                            return loss
                        grads_i_after = grad(proto_loss_i_after)((W, b))
                        # Flatten to single vector [param_size]
                        grad_vec_i_after = torch.cat([grads_i_after[0].flatten(), grads_i_after[1].flatten()])
                    else:
                        def proto_loss_i_after(W_inner):
                            out = F.linear(proto_feature_i, W_inner)
                            log_probs = F.log_softmax(out, dim=-1)
                            hard = idx.unsqueeze(-1)
                            loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                            return loss
                        grad_W_i_after = grad(proto_loss_i_after)(W)
                        grad_vec_i_after = grad_W_i_after.flatten()

                    grad_vecs_orig.append(grad_vec_i_orig)
                    grad_vecs_after.append(grad_vec_i_after)

                # Stack into matrices [num_prototypes, param_size]
                grad_vecs_orig = torch.stack(grad_vecs_orig)  # [K, D]
                grad_vecs_after = torch.stack(grad_vecs_after)  # [K, D]

                # Compute gradient change vectors [num_prototypes, param_size]
                delta_grads = grad_vecs_after - grad_vecs_orig  # [K, D]

                # Build coupling matrix: C = delta_grads @ delta_grads.T [K, K]
                coupling_matrix = torch.matmul(delta_grads, delta_grads.T)  # [K, K]

                # Compute score based on eigenvalue mode
                if self.pc_eigenvalue_mode == "participation_ratio":
                    # Participation Ratio: PR = (trace(C))^2 / trace(C^2)
                    # Measures effective number of participating prototypes
                    # CORRECTED: ID samples couple strongly with specific prototype (LOW PR)
                    #            OOD samples couple weakly with many prototypes (HIGH PR)
                    # ID: low PR (strongly coupled to specific prototype) → high score (NEGATE)
                    # OOD: high PR (weakly coupled to many prototypes) → low score (NEGATE)
                    trace_C = torch.trace(coupling_matrix)
                    trace_C2 = torch.trace(torch.matmul(coupling_matrix, coupling_matrix))
                    # Prevent division by zero
                    pr = (trace_C ** 2) / (trace_C2 + 1e-10)
                    score = -pr  # NEGATED: Lower PR = higher score (ID)

                elif self.pc_eigenvalue_mode == "direction_alignment":
                    # Gradient Direction Alignment
                    # Measures average cosine similarity between all pairs of gradient changes
                    # ID: high alignment (all prototypes change in similar directions) → high score
                    # OOD: low alignment (prototypes change in different directions) → low score

                    # Normalize gradient change vectors [K, D]
                    delta_grads_norm = delta_grads / (torch.norm(delta_grads, p=2, dim=1, keepdim=True) + 1e-10)

                    # Correlation matrix: normalized coupling [K, K]
                    correlation_matrix = torch.matmul(delta_grads_norm, delta_grads_norm.T)

                    # Average off-diagonal elements (exclude diagonal which is always 1.0)
                    # NOTE: Assume K > 1 (typically K = num_classes >= 2)
                    K = correlation_matrix.shape[0]
                    total_sum = correlation_matrix.sum()
                    diagonal_sum = torch.trace(correlation_matrix)
                    off_diagonal_sum = total_sum - diagonal_sum
                    avg_alignment = off_diagonal_sum / (K * (K - 1) + 1e-10)
                    score = avg_alignment  # Higher alignment = higher score (ID)

                elif self.pc_eigenvalue_mode == "spectral_entropy":
                    # Spectral Entropy: Entropy of eigenvalue distribution
                    # Measures uniformity of eigenvalue spectrum
                    # CORRECTED: ID samples have concentrated eigenvalues (LOW entropy)
                    #            OOD samples have uniform eigenvalues (HIGH entropy)
                    # ID: low entropy (largest eigenvalue dominant) → high score (NEGATE)
                    # OOD: high entropy (eigenvalues spread uniformly) → low score (NEGATE)

                    # Compute eigenvalues of coupling matrix (symmetric, so use eigvalsh)
                    eigenvalues = torch.linalg.eigvalsh(coupling_matrix)  # [K]

                    # Take absolute values and normalize to probability distribution
                    abs_eigs = torch.abs(eigenvalues)
                    eig_sum = abs_eigs.sum() + 1e-10  # Add epsilon to prevent division by zero

                    # Always compute (no data-dependent branching for vmap compatibility)
                    eig_probs = abs_eigs / eig_sum
                    # Compute Shannon entropy
                    entropy = -(eig_probs * torch.log(eig_probs + 1e-10)).sum()
                    score = -entropy  # NEGATED: Lower entropy = higher score (ID)

                elif self.pc_eigenvalue_mode == "gini_coefficient":
                    # Gini Coefficient: Measures gradient inequality (경제학적 불평등 측정)
                    # Inspired by economics: measures inequality in gradient norms
                    # ID: high Gini (gradient inequality, dominant prototype exists) → high score
                    # OOD: low Gini (gradient equality, all prototypes similar) → low score
                    # Formula: Gini = Σᵢ Σⱼ |gᵢ - gⱼ| / (2K² × mean(g))

                    # Compute gradient norms for each prototype
                    grad_norms = torch.norm(delta_grads, p=2, dim=1)  # [K]
                    K = grad_norms.shape[0]

                    # Compute pairwise absolute differences: |gᵢ - gⱼ| for all i, j
                    # diff_matrix[i,j] = |grad_norms[i] - grad_norms[j]|
                    diff_matrix = torch.abs(grad_norms.unsqueeze(0) - grad_norms.unsqueeze(1))  # [K, K]

                    # Gini coefficient formula
                    mean_grad = grad_norms.mean() + 1e-10  # Prevent division by zero
                    gini = diff_matrix.sum() / (2 * K * K * mean_grad)

                    score = gini  # Higher Gini = higher inequality = ID → high score

                elif self.pc_eigenvalue_mode == "principal_dominance":
                    # Principal Component Dominance: λ₁ / Σλᵢ
                    # Measures how much the largest eigenvalue dominates
                    # CORRECTED: ID samples have uniform eigenvalues (LOW dominance)
                    #            OOD samples have concentrated eigenvalues (HIGH dominance)
                    # ID: low dominance (eigenvalues spread uniformly) → high score (NEGATE)
                    # OOD: high dominance (largest eigenvalue dominates) → low score (NEGATE)
                    # Equivalent to "explained variance ratio" in PCA

                    # Compute eigenvalues (already available from coupling matrix)
                    eigenvalues = torch.linalg.eigvalsh(coupling_matrix)  # [K]

                    # Take absolute values
                    abs_eigs = torch.abs(eigenvalues)
                    largest_eig = abs_eigs.max()
                    eig_sum = abs_eigs.sum() + 1e-10  # Prevent division by zero

                    # Dominance ratio
                    dominance = largest_eig / eig_sum

                    score = -dominance  # NEGATED: Lower dominance = higher score (ID)

                elif self.pc_eigenvalue_mode == "max_mean_ratio":
                    # Max-Mean Ratio: max(||Δgᵢ||) / mean(||Δgᵢ||)
                    # Simple and intuitive measure of gradient inequality
                    # ID: high ratio (one prototype has much larger gradient) → high score
                    # OOD: low ratio (all prototypes have similar gradients) → low score

                    # Compute gradient norms for each prototype
                    grad_norms = torch.norm(delta_grads, p=2, dim=1)  # [K]

                    # Max-mean ratio
                    max_norm = grad_norms.max()
                    mean_norm = grad_norms.mean() + 1e-10  # Prevent division by zero
                    ratio = max_norm / mean_norm

                    score = ratio  # Higher ratio = ID → high score

                elif self.pc_eigenvalue_mode == "ipr":
                    # Inverse Participation Ratio (IPR)
                    # Physics: Anderson localization theory
                    # IPR = tr(C²) / (tr(C))² = 1/PR
                    # Range: [1/K, 1]
                    # ID: 1/K (delocalized, low IPR) → high score (NEGATE)
                    # OOD: 1 (localized, high IPR) → low score (NEGATE)

                    trace_C = torch.trace(coupling_matrix)
                    trace_C2 = torch.trace(torch.matmul(coupling_matrix, coupling_matrix))

                    # IPR = 1/PR
                    ipr = trace_C2 / ((trace_C ** 2) + 1e-10)

                    score = -ipr  # NEGATED: Lower IPR = higher score (ID)

                elif self.pc_eigenvalue_mode == "generalized_pr":
                    # Generalized Participation Ratio (GPR)
                    # Physics: Renyi entropy, Multifractal analysis
                    # PR_q = (Σλᵢ^q)^(1/(1-q))
                    # q is hyperparameter (self.pc_gpr_q)
                    # Special cases:
                    # - q=0: K (total number)
                    # - q=1: exp(Shannon entropy)
                    # - q=2: Standard PR (current implementation)
                    # - q→∞: 1/λ_max
                    # q < 2: more sensitive to small eigenvalues
                    # q > 2: more sensitive to large eigenvalues

                    q = self.pc_gpr_q
                    eigenvalues = torch.linalg.eigvalsh(coupling_matrix)  # [K]
                    abs_eigs = torch.abs(eigenvalues)

                    # Handle special cases for numerical stability
                    if abs(q - 1.0) < 1e-6:
                        # Special case q=1: exp(Shannon entropy)
                        probs = abs_eigs / (abs_eigs.sum() + 1e-10)
                        entropy = -(probs * torch.log(probs + 1e-10)).sum()
                        pr_q = torch.exp(entropy)
                    elif abs(q - 2.0) < 1e-6:
                        # Special case q=2: Standard PR (fast path)
                        trace_C = torch.trace(coupling_matrix)
                        trace_C2 = torch.trace(torch.matmul(coupling_matrix, coupling_matrix))
                        pr_q = (trace_C ** 2) / (trace_C2 + 1e-10)
                    else:
                        # General case
                        sum_q = (abs_eigs ** q).sum()
                        pr_q = sum_q ** (1.0 / (1.0 - q + 1e-10))

                    # ID: low PR_q → high score (NEGATE)
                    # OOD: high PR_q → low score (NEGATE)
                    score = -pr_q  # NEGATED

                elif self.pc_eigenvalue_mode == "magnitude_weighted_pr":
                    # Magnitude-Weighted Participation Ratio
                    # Combines eigenvalue MAGNITUDE (size) with DISTRIBUTION (shape)
                    # Key insight: ID has small + uniform eigenvalues, OOD has large + concentrated
                    # magnitude × distribution captures both signals

                    eigenvalues = torch.linalg.eigvalsh(coupling_matrix)
                    abs_eigs = torch.abs(eigenvalues)

                    # 1. Magnitude: Lp-norm of eigenvalues
                    q_mag = self.pc_mwpr_q
                    if torch.isinf(torch.tensor(q_mag)):
                        # q=inf: max eigenvalue
                        magnitude = abs_eigs.max()
                    else:
                        # General Lp norm: (Σλᵢ^q)^(1/q)
                        magnitude = (abs_eigs ** q_mag).sum() ** (1.0 / q_mag)

                    # 2. Distribution: Standard PR (shape only)
                    trace_C = torch.trace(coupling_matrix)
                    trace_C2 = torch.trace(torch.matmul(coupling_matrix, coupling_matrix))
                    pr = (trace_C ** 2) / (trace_C2 + 1e-10)

                    # 3. Combine magnitude and distribution
                    if self.pc_mwpr_combine_mode == "multiply":
                        # Multiplicative: amplifies ID/OOD difference
                        # OOD: large magnitude × high PR = very high
                        # ID: small magnitude × low PR = very low
                        score = -(magnitude * pr)  # NEGATED
                    elif self.pc_mwpr_combine_mode == "add":
                        # Additive: more stable
                        score = -(magnitude + pr)  # NEGATED
                    elif self.pc_mwpr_combine_mode == "log_add":
                        # Log-scale magnitude for better balance
                        score = -(torch.log(magnitude + 1.0) + pr)  # NEGATED
                    else:
                        score = -(magnitude * pr)  # Default: multiply, NEGATED

                elif self.pc_eigenvalue_mode == "spectrum_stats":
                    # Spectrum Statistics: Comprehensive multi-metric approach
                    # Combines multiple statistical measures of eigenvalue distribution
                    # Ensemble of magnitude and shape signals

                    eigenvalues = torch.linalg.eigvalsh(coupling_matrix)
                    abs_eigs = torch.abs(eigenvalues)

                    # 1. Total Energy: Σλᵢ (overall coupling strength)
                    total_energy = abs_eigs.sum()

                    # 2. Max Eigenvalue: max(λᵢ) (dominant mode strength)
                    max_eigenvalue = abs_eigs.max()

                    # 3. Gini Coefficient: gradient inequality
                    K = abs_eigs.shape[0]
                    diff_matrix = torch.abs(abs_eigs.unsqueeze(0) - abs_eigs.unsqueeze(1))
                    gini = diff_matrix.sum() / (2 * K * K * (abs_eigs.mean() + 1e-10))

                    # 4. Variance: spread of eigenvalues
                    variance = abs_eigs.var()

                    # Weighted combination
                    score = (self.pc_spectrum_w_total_energy * total_energy +
                            self.pc_spectrum_w_max_eig * max_eigenvalue +
                            self.pc_spectrum_w_gini * gini +
                            self.pc_spectrum_w_variance * variance)

                elif self.pc_eigenvalue_mode == "dual_metric":
                    # Dual Metric: Magnitude + Shape with tunable weights
                    # Simple and interpretable two-component model
                    # α × magnitude + β × shape

                    eigenvalues = torch.linalg.eigvalsh(coupling_matrix)
                    abs_eigs = torch.abs(eigenvalues)

                    # Magnitude score: total trace (coupling strength)
                    magnitude_score = abs_eigs.sum()

                    # Shape score: negative PR (distribution measure)
                    trace_C = torch.trace(coupling_matrix)
                    trace_C2 = torch.trace(torch.matmul(coupling_matrix, coupling_matrix))
                    pr = (trace_C ** 2) / (trace_C2 + 1e-10)
                    shape_score = -pr  # Negated

                    # Weighted combination
                    score = self.pc_dual_alpha * magnitude_score + self.pc_dual_beta * shape_score

                elif self.pc_eigenvalue_mode == "log_magnitude_pr":
                    # Log-Magnitude + PR: Additive combination with log-scaled magnitude
                    # More stable than multiplicative, compresses magnitude range

                    eigenvalues = torch.linalg.eigvalsh(coupling_matrix)
                    abs_eigs = torch.abs(eigenvalues)

                    # Log-magnitude: log(Σλᵢ)
                    log_magnitude = torch.log(abs_eigs.sum() + 1e-8)

                    # PR score (negated)
                    trace_C = torch.trace(coupling_matrix)
                    trace_C2 = torch.trace(torch.matmul(coupling_matrix, coupling_matrix))
                    pr = (trace_C ** 2) / (trace_C2 + 1e-10)
                    pr_score = -pr

                    # Additive combination
                    score = log_magnitude + pr_score

                elif self.pc_eigenvalue_mode == "effective_rank":
                    # Effective Rank
                    # Physics: Information theory + Statistical mechanics
                    # ER = exp(H) = exp(-Σpᵢ·log(pᵢ))
                    # Range: [1, K]
                    # ID: K (high ER, uniform eigenvalues) → high score
                    # OOD: 1 (low ER, concentrated eigenvalues) → low score
                    # Interpretation: "Effective number of eigenvalues" (geometric mean)

                    eigenvalues = torch.linalg.eigvalsh(coupling_matrix)  # [K]
                    abs_eigs = torch.abs(eigenvalues)

                    # Normalize to probability distribution
                    probs = abs_eigs / (abs_eigs.sum() + 1e-10)

                    # Compute Shannon entropy
                    entropy = -(probs * torch.log(probs + 1e-10)).sum()

                    # Effective rank = exp(entropy)
                    eff_rank = torch.exp(entropy)

                    score = eff_rank  # Higher ER = higher score (ID)

                elif self.pc_eigenvalue_mode == "quantum_purity":
                    # Quantum Purity
                    # Physics: Quantum mechanics (density matrix purity)
                    # Purity = tr(ρ²) = Σ(λᵢ/Σλⱼ)²
                    # Range: [1/K, 1]
                    # ID: 1/K (maximally mixed state) → high score (NEGATE)
                    # OOD: 1 (pure state) → low score (NEGATE)
                    # Interpretation: Quantum mechanical "mixedness" measure

                    eigenvalues = torch.linalg.eigvalsh(coupling_matrix)  # [K]
                    abs_eigs = torch.abs(eigenvalues)

                    # Normalize to probability distribution (density matrix eigenvalues)
                    probs = abs_eigs / (abs_eigs.sum() + 1e-10)

                    # Purity = Σpᵢ²
                    purity = (probs ** 2).sum()

                    score = -purity  # NEGATED: Lower purity (mixed) = higher score (ID)

                elif self.pc_eigenvalue_mode == "concentration":
                    # Legacy mode: Use variance of gradient norms (for backward compatibility)
                    # ID: low variance (diffuse) → high score (NEGATE)
                    # OOD: high variance (concentrated) → low score (NEGATE)
                    grad_norms = torch.norm(delta_grads, p=2, dim=1)  # [K]
                    mean_norm = grad_norms.mean()
                    variance = ((grad_norms - mean_norm) ** 2).mean()
                    score = -variance  # NEGATED

                elif self.pc_eigenvalue_mode == "entropy":
                    # Legacy mode: Shannon entropy of gradient norm distribution (for backward compatibility)
                    # ID: high entropy (diffuse) → high score
                    # OOD: low entropy (concentrated) → low score
                    grad_norms = torch.norm(delta_grads, p=2, dim=1)  # [K]
                    abs_norms = torch.abs(grad_norms)
                    probs = abs_norms / (abs_norms.sum() + 1e-10)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum()
                    score = entropy  # NO negation: higher entropy = higher score

                else:
                    # Default: use participation_ratio
                    trace_C = torch.trace(coupling_matrix)
                    trace_C2 = torch.trace(torch.matmul(coupling_matrix, coupling_matrix))
                    pr = (trace_C ** 2) / (trace_C2 + 1e-10)
                    score = pr
            else:
                # Fallback: use delta_energy
                score = -dE
        elif self.score_type == "gradient_alignment":
            # Sample-Prototype Gradient Alignment
            # Measures cosine similarity between sample gradient and prototype gradient
            # ID: high alignment (sample and prototype gradients point in same direction)
            # OOD: low alignment (different directions)
            if prototype_tensor is not None:
                pseudo_class = target.argmax(dim=-1)

                # Get predicted class prototype
                one_hot = F.one_hot(pseudo_class, num_classes=prototype_tensor.shape[0]).float()
                proto_feature = torch.matmul(one_hot, prototype_tensor)

                # Compute gradient on SAMPLE feature (after unlearning)
                if b is not None:
                    def sample_loss_fn(params):
                        W_inner, b_inner = params
                        out = F.linear(feature, W_inner, b_inner)
                        log_probs = F.log_softmax(out, dim=-1)
                        hard = pseudo_class.unsqueeze(-1)
                        loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                        return loss
                    grads_sample = grad(sample_loss_fn)((W, b))
                    # Flatten gradients
                    grad_sample_flat = torch.cat([grads_sample[0].flatten(), grads_sample[1].flatten()])
                else:
                    def sample_loss_fn(W_inner):
                        out = F.linear(feature, W_inner)
                        log_probs = F.log_softmax(out, dim=-1)
                        hard = pseudo_class.unsqueeze(-1)
                        loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                        return loss
                    grad_sample = grad(sample_loss_fn)(W)
                    grad_sample_flat = grad_sample.flatten()

                # Compute gradient on PROTOTYPE feature (after unlearning)
                if b is not None:
                    def proto_loss_fn(params):
                        W_inner, b_inner = params
                        out = F.linear(proto_feature, W_inner, b_inner)
                        log_probs = F.log_softmax(out, dim=-1)
                        hard = pseudo_class.unsqueeze(-1)
                        loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                        return loss
                    grads_proto = grad(proto_loss_fn)((W, b))
                    # Flatten gradients
                    grad_proto_flat = torch.cat([grads_proto[0].flatten(), grads_proto[1].flatten()])
                else:
                    def proto_loss_fn(W_inner):
                        out = F.linear(proto_feature, W_inner)
                        log_probs = F.log_softmax(out, dim=-1)
                        hard = pseudo_class.unsqueeze(-1)
                        loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                        return loss
                    grad_proto = grad(proto_loss_fn)(W)
                    grad_proto_flat = grad_proto.flatten()

                # Compute cosine similarity
                if self.ga_normalize_grads:
                    # L2 normalize
                    grad_sample_norm = grad_sample_flat / (torch.norm(grad_sample_flat, p=2) + 1e-10)
                    grad_proto_norm = grad_proto_flat / (torch.norm(grad_proto_flat, p=2) + 1e-10)
                    cosine_sim = torch.dot(grad_sample_norm, grad_proto_norm)
                else:
                    # Raw dot product
                    cosine_sim = torch.dot(grad_sample_flat, grad_proto_flat)
                    cosine_sim = cosine_sim / (torch.norm(grad_sample_flat, p=2) * torch.norm(grad_proto_flat, p=2) + 1e-10)

                # Apply absolute if configured
                if self.ga_use_absolute:
                    score = torch.abs(cosine_sim)
                else:
                    score = cosine_sim

                # ID samples should have HIGH alignment → high score
                # OOD samples should have LOW alignment → low score
                # No negation needed
            else:
                # Fallback: use delta_energy
                score = -dE
        elif self.score_type == "confidence_entropy_combo":
            # Confidence Drop × Entropy Change
            # Measures output-space changes (not weight-space)
            # CORRECTED: ID samples remain confident (small changes), OOD samples lose confidence
            # ID: small conf_drop × small entropy_increase → high score (NEGATE)
            # OOD: large conf_drop × large entropy_increase → low score (NEGATE)

            # Stabilize logits
            logits_orig_stable = logits_orig - logits_orig.max(dim=-1, keepdim=True).values
            logits_after_stable = logits_after - logits_after.max(dim=-1, keepdim=True).values

            # Compute softmax
            softmax_orig = F.softmax(logits_orig_stable, dim=-1)
            softmax_after = F.softmax(logits_after_stable, dim=-1)

            # Confidence drop
            conf_orig = softmax_orig.max(dim=-1).values
            conf_after = softmax_after.max(dim=-1).values
            conf_drop = conf_orig - conf_after  # Positive if confidence decreased

            # Entropy change
            entropy_orig = -(softmax_orig * torch.log(softmax_orig + 1e-10)).sum(dim=-1)
            entropy_after = -(softmax_after * torch.log(softmax_after + 1e-10)).sum(dim=-1)
            entropy_increase = entropy_after - entropy_orig  # Positive if entropy increased

            # Combined score: multiplicative
            # ID: small conf_drop × small entropy_increase → high score (NEGATE)
            # OOD: large conf_drop × large entropy_increase → low score (NEGATE)
            score = -(conf_drop * entropy_increase)  # NEGATED
        elif self.score_type in {"gradnorm", "g"}:
            # gradnorm 후처리 - 업데이트된 파라미터로 계산
            if b is not None:
                def loss_fn_after(params):
                    W_inner, b_inner = params
                    out = F.linear(feature, W_inner, b_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = target.argmax(dim=-1, keepdim=True)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss

                grads_after = grad(loss_fn_after)((W, b))
                gradnorm_val = torch.abs(grads_after[0]).sum() + torch.abs(grads_after[1]).sum()
            else:
                def loss_fn_after(W_inner):
                    out = F.linear(feature, W_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = target.argmax(dim=-1, keepdim=True)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss

                grad_W_after = grad(loss_fn_after)(W)
                gradnorm_val = torch.abs(grad_W_after).sum()

            score = gradnorm_val - gradnorm_o if gradnorm_o is not None else gradnorm_val  # OOD should have higher gradnorm after unlearning
        elif self.score_type == "grad_ratio":
            # Gradient magnitude ratio
            if b is not None:
                def loss_fn_after(params):
                    W_inner, b_inner = params
                    out = F.linear(feature, W_inner, b_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = target.argmax(dim=-1, keepdim=True)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss

                grads_after = grad(loss_fn_after)((W, b))
                gradnorm_after = torch.abs(grads_after[0]).sum() + torch.abs(grads_after[1]).sum()
            else:
                def loss_fn_after(W_inner):
                    out = F.linear(feature, W_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = target.argmax(dim=-1, keepdim=True)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss

                grad_W_after = grad(loss_fn_after)(W)
                gradnorm_after = torch.abs(grad_W_after).sum()

            # vmap 호환: torch.where 사용
            if gradnorm_o is not None:
                ratio = gradnorm_after / (gradnorm_o + 1e-10)
                score = ratio - 1.0  # OOD has higher ratio (less gradient reduction)
            else:
                score = gradnorm_after  # gradnorm_o가 없으면 after만 사용
        elif self.score_type == "energy_curvature":
            # Energy Trajectory Curvature
            # ID: smooth energy decrease → low curvature (high score after negation)
            # OOD: irregular energy changes → high curvature (low score after negation)
            if traj_step_count > 1:
                # Mean absolute curvature
                mean_curvature = sum_abs_ddE / max(1, traj_step_count - 1)
                # Curvature variance (for additional signal)
                curvature_variance = sum_ddE_sq / max(1, traj_step_count - 1) - mean_curvature ** 2
                # Lower curvature = smoother = ID → negate for higher score
                score = -(mean_curvature + 0.1 * curvature_variance)
            else:
                # Fallback to delta_energy if only 1 step
                score = -dE
        elif self.score_type == "weight_convergence":
            # Weight Update Convergence Pattern
            # ID: consistent direction, decreasing magnitude → high score
            # OOD: inconsistent direction, stable magnitude → low score
            if traj_step_count > 1:
                # Convergence rate: (first_norm - final_norm) / first_norm
                if first_delta_norm is not None and first_delta_norm > 1e-10:
                    convergence_rate = (first_delta_norm - delta_norm_curr) / first_delta_norm
                else:
                    convergence_rate = torch.tensor(0.0, device=device)

                # Direction consistency: average cosine similarity
                if traj_step_count > 1:
                    direction_consistency = sum_direction_cos / max(1, traj_step_count - 1)
                else:
                    direction_consistency = torch.tensor(0.0, device=device)

                # Combine: higher convergence + higher consistency = ID
                score = 0.5 * convergence_rate + 0.5 * direction_consistency
            else:
                # Fallback to delta_energy
                score = -dE
        elif self.score_type == "gradient_decay":
            # Gradient Magnitude Decay Rate
            # ID: fast gradient decay (rapid convergence) → high decay rate → high score
            # OOD: slow gradient decay (flat landscape) → low decay rate → low score
            if traj_step_count > 0 and g_0 > 1e-10:
                # Exponential decay rate: λ ≈ (log(g_0) - log(g_final)) / num_steps
                decay_rate = (torch.log(g_0 + 1e-10) - torch.log(g_final + 1e-10)) / max(1, traj_step_count)
                score = decay_rate  # Higher decay = faster convergence = ID
            else:
                # Fallback to delta_energy
                score = -dE
        elif self.score_type == "coupling_evolution":
            # Prototype Coupling Evolution (FULL VERSION)
            # Measures how coupling metric evolves across trajectory
            # Uses the SAME eigenvalue analysis as prototype_coupling, but tracks it over time
            # ID: stable coupling pattern → low variance, consistent metric
            # OOD: unstable coupling pattern → high variance, erratic metric

            if coupling_step_count > 1:
                # Compute statistics of coupling metric trajectory
                mean_coupling = sum_coupling_metric / coupling_step_count
                var_coupling = sum_coupling_metric_sq / coupling_step_count - mean_coupling ** 2

                # Step-to-step instability
                mean_coupling_diff = sum_abs_coupling_diff / max(1, coupling_step_count - 1)

                # Combined score based on trajectory stability
                # Interpretation depends on eigenvalue mode:
                # - participation_ratio, spectral_entropy: ID has LOW and STABLE values
                # - gini_coefficient, max_mean_ratio: ID has HIGH and STABLE values

                # Use mode-specific logic
                if self.pc_eigenvalue_mode in ["participation_ratio", "spectral_entropy", "ipr"]:
                    # Lower metric = ID, so stable LOW values → high score
                    # Use: -mean_coupling (lower = better) AND -var_coupling (stable = better)
                    score = -(mean_coupling + 0.5 * var_coupling + 0.3 * mean_coupling_diff)
                elif self.pc_eigenvalue_mode in ["gini_coefficient", "max_mean_ratio"]:
                    # Higher metric = ID, so stable HIGH values → high score
                    # Use: +mean_coupling (higher = better) AND -var_coupling (stable = better)
                    score = mean_coupling - (0.5 * var_coupling + 0.3 * mean_coupling_diff)
                else:
                    # Default: assume lower metric = ID (like PR)
                    score = -(mean_coupling + 0.5 * var_coupling + 0.3 * mean_coupling_diff)

            else:
                # Fallback to delta_energy if only 1 step or coupling not computed
                score = -dE
        elif self.score_type == "trajectory_combo":
            # Combination of all trajectory-based signals
            # Weighted sum of energy_curvature, weight_convergence, gradient_decay, coupling_evolution

            score_components = []

            # [1] Energy curvature
            if traj_step_count > 1:
                mean_curvature = sum_abs_ddE / max(1, traj_step_count - 1)
                curvature_variance = sum_ddE_sq / max(1, traj_step_count - 1) - mean_curvature ** 2
                energy_score = -(mean_curvature + 0.1 * curvature_variance)
            else:
                energy_score = -dE
            score_components.append(self.traj_w_energy * energy_score)

            # [2] Weight convergence
            if traj_step_count > 1:
                if first_delta_norm is not None and first_delta_norm > 1e-10:
                    convergence_rate = (first_delta_norm - delta_norm_curr) / first_delta_norm
                else:
                    convergence_rate = torch.tensor(0.0, device=device)

                if traj_step_count > 1:
                    direction_consistency = sum_direction_cos / max(1, traj_step_count - 1)
                else:
                    direction_consistency = torch.tensor(0.0, device=device)

                weight_score = 0.5 * convergence_rate + 0.5 * direction_consistency
            else:
                weight_score = -dE
            score_components.append(self.traj_w_weight * weight_score)

            # [3] Gradient decay
            if traj_step_count > 0 and g_0 > 1e-10:
                decay_rate = (torch.log(g_0 + 1e-10) - torch.log(g_final + 1e-10)) / max(1, traj_step_count)
                gradient_score = decay_rate
            else:
                gradient_score = -dE
            score_components.append(self.traj_w_gradient * gradient_score)

            # [4] Coupling evolution (FULL VERSION)
            if coupling_step_count > 1:
                # Use FULL coupling metric trajectory (computed in loop)
                mean_coupling = sum_coupling_metric / coupling_step_count
                var_coupling = sum_coupling_metric_sq / coupling_step_count - mean_coupling ** 2
                mean_coupling_diff = sum_abs_coupling_diff / max(1, coupling_step_count - 1)

                # Mode-specific scoring (same logic as coupling_evolution)
                if self.pc_eigenvalue_mode in ["participation_ratio", "spectral_entropy", "ipr"]:
                    coupling_score = -(mean_coupling + 0.5 * var_coupling + 0.3 * mean_coupling_diff)
                elif self.pc_eigenvalue_mode in ["gini_coefficient", "max_mean_ratio"]:
                    coupling_score = mean_coupling - (0.5 * var_coupling + 0.3 * mean_coupling_diff)
                else:
                    coupling_score = -(mean_coupling + 0.5 * var_coupling + 0.3 * mean_coupling_diff)
            else:
                coupling_score = -dE
            score_components.append(self.traj_w_coupling * coupling_score)

            # Combine all components
            score = sum(score_components)
        else:
            # combo - delta_energy, gradnorm, grad_ratio의 가중 합
            # CORRECTED: dE is negative (ID more negative than OOD), so negate to get ID > OOD
            score = self.w_dE * (-dE)

            # Gradnorm 기반 메트릭
            if self.use_gradnorm and gradnorm_o is not None:
                if b is not None:
                    def loss_fn_after(params):
                        W_inner, b_inner = params
                        out = F.linear(feature, W_inner, b_inner)
                        log_probs = F.log_softmax(out, dim=-1)
                        hard = target.argmax(dim=-1, keepdim=True)
                        loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                        return loss

                    grads_after = grad(loss_fn_after)((W, b))
                    gradnorm_after = torch.abs(grads_after[0]).sum() + torch.abs(grads_after[1]).sum()
                else:
                    def loss_fn_after(W_inner):
                        out = F.linear(feature, W_inner)
                        log_probs = F.log_softmax(out, dim=-1)
                        hard = target.argmax(dim=-1, keepdim=True)
                        loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                        return loss

                    grad_W_after = grad(loss_fn_after)(W)
                    gradnorm_after = torch.abs(grad_W_after).sum()

                # Gradnorm difference - OOD should have higher gradnorm after unlearning
                gradnorm_contrib = gradnorm_after - gradnorm_o
                score = score + (self.w_G * gradnorm_contrib)

                # Gradient ratio (vmap 호환) - OOD has higher ratio
                ratio = gradnorm_after / (gradnorm_o + 1e-10)
                ratio_contrib = ratio - 1.0
                score = score + (self.w_ratio * ratio_contrib)

        # Phase 3.2: Add L2 regularization penalty (optional)
        if self.l2_penalty > 0.0 and weight_shift_norm is not None:
            # Penalize large weight shifts
            # ID samples: small shift → small penalty
            # OOD samples: large shift → large penalty
            # Since we want ID > OOD, we SUBTRACT the penalty (higher shift = lower score)
            l2_term = self.l2_penalty * weight_shift_norm
            score = score - l2_term

        # NaN이나 Inf가 있으면 0으로 대체
        score = torch.where(torch.isnan(score) | torch.isinf(score),
                           torch.tensor(0.0, device=device, dtype=score.dtype),
                           score)

        return pred, score

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        """Setup phase: compute class prototypes from ID validation data

        Computes median feature for each class and stores as prototypes.
        Prototypes are cached to disk and reused if available.
        """
        from tqdm import tqdm
        import openood.utils.comm as comm
        import os

        if not id_loader_dict or 'val' not in id_loader_dict:
            print("Warning: No ID validation data available for prototype computation")
            self.class_prototypes = None
            return

        # Extract a sample to determine feature dimension
        id_loader = id_loader_dict['val']
        sample_batch = next(iter(id_loader))
        sample_data = sample_batch['data'].cuda()

        with torch.no_grad():
            try:
                _, sample_features = net(sample_data, return_feature=True)
            except:
                if hasattr(net, 'forward_features'):
                    sample_features = net.forward_features(sample_data)
                else:
                    sample_features = net(sample_data)

        feature_dim = sample_features.shape[-1]

        # Define prototype cache path
        dataset_name = self.config.dataset.name
        cache_dir = "data/prototypes"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{dataset_name}_dim{feature_dim}_prototypes.pt")

        # Try to load cached prototypes
        if os.path.exists(cache_file):
            print(f"Loading cached prototypes from {cache_file}")
            try:
                cached_data = torch.load(cache_file)
                if (cached_data['num_classes'] == self.num_classes and
                    cached_data['feature_dim'] == feature_dim):
                    self.class_prototypes = cached_data['prototypes']
                    # Load global prototype if available
                    if 'global_prototype' in cached_data and cached_data['global_prototype'] is not None:
                        self.global_prototype = cached_data['global_prototype'].cuda()
                        print(f"✓ Loaded {len(self.class_prototypes)} class prototypes + global prototype from cache")
                    else:
                        self.global_prototype = None
                        print(f"✓ Loaded {len(self.class_prototypes)} class prototypes from cache (no global prototype)")
                    for cls_idx, proto in self.class_prototypes.items():
                        print(f"  Class {cls_idx}: shape {proto.shape}")
                    return
                else:
                    print(f"  Cache mismatch (num_classes: {cached_data['num_classes']} vs {self.num_classes}, "
                          f"feature_dim: {cached_data['feature_dim']} vs {feature_dim}). Recomputing...")
            except Exception as e:
                print(f"  Error loading cache: {e}. Recomputing...")

        print("Computing class prototypes from ID validation data...")

        # Collect features for each class
        class_features = {i: [] for i in range(self.num_classes)}

        net.eval()
        with torch.no_grad():
            for batch in tqdm(id_loader,
                            desc="Extracting features",
                            disable=not comm.is_main_process()):
                data = batch['data'].cuda()
                labels = batch['label'].cuda()

                # Extract features
                try:
                    _, features = net(data, return_feature=True)
                except:
                    # Fallback if return_feature not supported
                    if hasattr(net, 'forward_features'):
                        features = net.forward_features(data)
                    else:
                        features = net(data)

                # Group by class
                for i in range(len(labels)):
                    cls_idx = labels[i].item()
                    if cls_idx < self.num_classes:
                        class_features[cls_idx].append(features[i].cpu())

        # Compute median feature for each class
        self.class_prototypes = {}
        all_features_list = []  # For global prototype

        for cls_idx in range(self.num_classes):
            if len(class_features[cls_idx]) > 0:
                feats = torch.stack(class_features[cls_idx])  # [N, feature_dim]
                # Compute median along dimension 0 (across samples)
                self.class_prototypes[cls_idx] = torch.median(feats, dim=0).values.cuda()
                print(f"  Class {cls_idx}: {len(class_features[cls_idx])} samples, "
                      f"prototype shape: {self.class_prototypes[cls_idx].shape}")
                # Collect for global prototype
                all_features_list.extend(class_features[cls_idx])
            else:
                print(f"  Warning: No samples found for class {cls_idx}")
                # Use zero vector as fallback
                if cls_idx > 0 and (cls_idx - 1) in self.class_prototypes:
                    feature_dim_fallback = self.class_prototypes[cls_idx - 1].shape[0]
                else:
                    feature_dim_fallback = feature_dim  # Use detected feature_dim
                self.class_prototypes[cls_idx] = torch.zeros(feature_dim_fallback).cuda()

        # Compute global ID prototype (median of all ID features)
        if len(all_features_list) > 0:
            all_features = torch.stack(all_features_list)  # [Total_N, feature_dim]
            self.global_prototype = torch.median(all_features, dim=0).values.cuda()
            print(f"  Global ID prototype computed from {len(all_features_list)} total samples")
        else:
            self.global_prototype = None
            print(f"  Warning: No features for global prototype")

        print(f"Prototype computation complete: {len(self.class_prototypes)} class prototypes + 1 global prototype")

        # Save prototypes to cache
        try:
            # Move prototypes to CPU for saving
            prototypes_cpu = {k: v.cpu() for k, v in self.class_prototypes.items()}
            global_proto_cpu = self.global_prototype.cpu() if self.global_prototype is not None else None
            cache_data = {
                'num_classes': self.num_classes,
                'feature_dim': feature_dim,
                'prototypes': prototypes_cpu,
                'global_prototype': global_proto_cpu,
                'dataset_name': dataset_name
            }
            torch.save(cache_data, cache_file)
            print(f"✓ Saved prototypes to {cache_file}")
            # Move back to GPU
            self.class_prototypes = {k: v.cuda() for k, v in prototypes_cpu.items()}
            if global_proto_cpu is not None:
                self.global_prototype = global_proto_cpu.cuda()
        except Exception as e:
            print(f"Warning: Failed to save prototypes to cache: {e}")

        # Compute statistics for adaptive normalization (feature_aware advanced mode)
        if self.score_type == "feature_aware" and (
            self.fa_use_adaptive_norm or self.fa_mode in ["adaptive_norm", "weighted", "angular", "nonlinear", "full"]
        ):
            print("Computing statistics for feature_aware adaptive normalization...")
            self._compute_feature_aware_stats(net, id_loader_dict)

        # Compute feature importance for weighted_l1 distance metric
        if self.fa_distance_metric == "weighted_l1":
            print(f"Computing feature importance for weighted_l1 (mode: {self.fa_weighted_l1_importance_mode})...")
            self.feature_importance = self._compute_feature_importance(
                net, id_loader_dict, self.fa_weighted_l1_importance_mode
            )

        # Compute RBF sigma for rbf distance metric
        if self.fa_distance_metric == "rbf" and isinstance(self.fa_rbf_sigma, str):
            print(f"Computing RBF sigma (mode: {self.fa_rbf_sigma})...")
            self.fa_rbf_sigma_value = self._compute_rbf_sigma(
                net, id_loader_dict, self.fa_rbf_sigma
            )
            print(f"✓ RBF sigma computed: {self.fa_rbf_sigma_value:.4f}")

    def _compute_feature_aware_stats(self, net: nn.Module, id_loader_dict):
        """Compute statistics for adaptive normalization in feature_aware mode

        Computes mean and std for:
        - feature_norm
        - distance_to_prototype
        - weight_shift_norm (estimated)
        """
        if not id_loader_dict or 'val' not in id_loader_dict:
            print("  Warning: No ID validation data for statistics")
            return

        id_loader = id_loader_dict['val']
        net.eval()

        # Get FC parameters
        fc_weight, fc_bias = self._get_fc_params(net)
        if fc_weight is None:
            print("  Warning: Cannot extract FC parameters")
            return

        # Lists to collect values
        feature_norms = []
        distances = []
        weight_shifts = []

        with torch.no_grad():
            for batch in id_loader:
                data = batch['data'].cuda()
                labels = batch['label'].cuda()

                # Extract features
                try:
                    _, features = net(data, return_feature=True)
                except:
                    if hasattr(net, 'forward_features'):
                        features = net.forward_features(data)
                    else:
                        features = net(data)

                # Process each sample
                for i in range(features.shape[0]):
                    feature = features[i]
                    label = labels[i].item()

                    # Feature norm
                    f_norm = torch.norm(feature, p=2).item()
                    feature_norms.append(f_norm)

                    # Distance to prototype (using configurable metric)
                    if self.class_prototypes and label in self.class_prototypes:
                        proto = self.class_prototypes[label]
                        dist = self._compute_distance(feature, proto).item()
                        distances.append(dist)

                    # Estimate weight shift (we can't compute actual unlearning here,
                    # so use a heuristic based on gradient magnitude)
                    # Simplified: use feature norm as proxy
                    weight_shifts.append(f_norm * 0.01)  # Rough estimate

        # Compute statistics
        if feature_norms:
            self.fa_stats["feature_norm_mean"] = float(np.mean(feature_norms))
            self.fa_stats["feature_norm_std"] = float(np.std(feature_norms))
            print(f"  Feature norm: μ={self.fa_stats['feature_norm_mean']:.2f}, σ={self.fa_stats['feature_norm_std']:.2f}")

        if distances:
            self.fa_stats["distance_mean"] = float(np.mean(distances))
            self.fa_stats["distance_std"] = float(np.std(distances))
            print(f"  Distance: μ={self.fa_stats['distance_mean']:.2f}, σ={self.fa_stats['distance_std']:.2f}")

        if weight_shifts:
            self.fa_stats["weight_shift_mean"] = float(np.mean(weight_shifts))
            self.fa_stats["weight_shift_std"] = float(np.std(weight_shifts))
            print(f"  Weight shift (estimated): μ={self.fa_stats['weight_shift_mean']:.4f}, σ={self.fa_stats['weight_shift_std']:.4f}")

        print("  ✓ Statistics computed for adaptive normalization")

    def inference(self, net: nn.Module, data_loader, progress: bool = True):
        """Feature 추출 후 FC만 업데이트하는 방식"""
        from tqdm import tqdm
        import openood.utils.comm as comm

        pred_list, conf_list, label_list = [], [], []

        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()

            # 배치 전체를 처리
            pred, conf = self.postprocess(net, data)
            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # numpy로 변환
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """Feature 추출 후 FC만 업데이트하는 방식

        Args:
            net: 원본 네트워크
            data: 입력 데이터 [B, C, H, W]

        Returns:
            pred: 예측 [B]
            score: 스코어 [B]
        """
        device = data.device
        batch_size = data.shape[0]

        # Feature 추출
        with torch.no_grad():
            # return_feature=True로 feature 추출
            try:
                _, features = net(data, return_feature=True)
            except:
                # return_feature를 지원하지 않는 경우 forward_features 시도
                features = net.forward_features(data) if hasattr(net, 'forward_features') else net(data)

        # FC 파라미터 추출
        fc_weight, fc_bias = self._get_fc_params(net)

        # 배치 크기만큼 FC 파라미터 복제
        fc_weights = fc_weight.unsqueeze(0).expand(batch_size, *fc_weight.shape)
        if fc_bias is not None:
            fc_biases = fc_bias.unsqueeze(0).expand(batch_size, *fc_bias.shape)
        else:
            fc_biases = None

        # 프로토타입 텐서 준비 (setup()에서 계산된 경우)
        prototype_tensor = None
        global_proto = None
        if hasattr(self, 'class_prototypes') and self.class_prototypes is not None:
            # 모든 클래스의 프로토타입을 텐서로 변환 [num_classes, feature_dim]
            prototype_list = [self.class_prototypes[i] for i in range(self.num_classes)]
            prototype_tensor = torch.stack(prototype_list).to(device)

        # Global prototype 준비
        if hasattr(self, 'global_prototype') and self.global_prototype is not None:
            global_proto = self.global_prototype.to(device)

        # Angular mode validation: ensure prototypes are available
        if (self.score_type == "feature_aware" and
            (self.fa_use_cosine or self.fa_mode in ["angular", "full"]) and
            prototype_tensor is None):
            raise RuntimeError(
                "Angular mode (fa_use_cosine=True or fa_mode='angular'/'full') requires class prototypes, "
                "but they were not computed during setup(). Please ensure setup() is called with valid ID data."
            )

        # vmap을 사용하여 배치 병렬화
        if fc_biases is not None:
            vmapped_fn = vmap(
                lambda w, b, f, proto, g_proto: self._single_sample_unlearn(w, b, f, proto, g_proto, device),
                in_dims=(0, 0, 0, None, None),  # fc_weight, fc_bias, feature, prototype_tensor (broadcast), global_proto (broadcast)
                out_dims=(0, 0)  # pred, score
            )
            preds, scores = vmapped_fn(fc_weights, fc_biases, features, prototype_tensor, global_proto)
        else:
            vmapped_fn = vmap(
                lambda w, f, proto, g_proto: self._single_sample_unlearn(w, None, f, proto, g_proto, device),
                in_dims=(0, 0, None, None),  # fc_weight, feature, prototype_tensor (broadcast), global_proto (broadcast)
                out_dims=(0, 0)  # pred, score
            )
            preds, scores = vmapped_fn(fc_weights, features, prototype_tensor, global_proto)

        return preds.squeeze(-1), scores
