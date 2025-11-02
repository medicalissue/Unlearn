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

        # Feature-aware gradient options
        self.use_feature_grad = bool(self.args.get("use_feature_grad", True))  # Enable feature gradient computation

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
            dist_to_prototype_orig = torch.norm(feature - proto_feature, p=2)

            # Global prototype까지의 거리
            if global_proto is not None:
                dist_to_global_orig = torch.norm(feature - global_proto, p=2)

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

        # Unlearning 수행
        target_final = target  # 마지막 target 저장용
        for step in range(max(1, self.num_steps)):
            # 매 스텝마다 target 재계산 옵션
            if self.recompute_target and step > 0:
                # 현재 파라미터로 로짓 재계산
                with torch.no_grad():
                    if b is not None:
                        logits_current = F.linear(feature, W, b)
                    else:
                        logits_current = F.linear(feature, W)
                    target = self._pseudo_target(logits_current).to(device=device, dtype=logits_current.dtype)
                    target_final = target  # 업데이트된 target 저장

            if b is not None:
                def loss_fn_unlearn(params):
                    W_inner, b_inner = params
                    out = F.linear(feature, W_inner, b_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = target.argmax(dim=-1, keepdim=True)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss

                grads = grad(loss_fn_unlearn)((W, b))

                # 파라미터 업데이트
                if self.unlearn_mode == "fisher":
                    # Fisher-weighted update
                    W = W + self.eta * grads[0] * (fisher_W + self.fisher_damping)
                    b = b + self.eta * grads[1] * (fisher_b + self.fisher_damping)
                else:
                    # Gradient ascent
                    W = W + self.eta * grads[0]
                    b = b + self.eta * grads[1]
            else:
                def loss_fn_unlearn(W_inner):
                    out = F.linear(feature, W_inner)
                    log_probs = F.log_softmax(out, dim=-1)
                    hard = target.argmax(dim=-1, keepdim=True)
                    loss = -torch.gather(log_probs, -1, hard).squeeze(-1)
                    return loss

                grad_W = grad(loss_fn_unlearn)(W)

                # 파라미터 업데이트
                if self.unlearn_mode == "fisher":
                    # Fisher-weighted update
                    W = W + self.eta * grad_W / (fisher_W + self.fisher_damping)
                else:
                    # Gradient ascent
                    W = W + self.eta * grad_W

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
            score = dE  # OOD samples have larger dE (energy increase), should get higher scores
        elif self.score_type == "feature_aware":
            # Feature geometry + FC dynamics
            # CORRECTED: ID samples have HIGHER scores (smaller distance, but we negate)
            # ID: small distance to prototype, small weight shift → low raw score → high after negation
            # OOD: large distance to prototype, large weight shift (paradoxically) → high raw score → low after negation
            # NOTE: Gradient ascent affects confident ID predictions MORE than uncertain OOD
            if weight_shift_norm is not None and dist_to_prototype_orig is not None:
                # Normalize each component for better scaling
                raw_score = (
                    (feature_norm / 100.0) *  # Typical feature norm ~100
                    dist_to_prototype_orig *   # Distance to prototype
                    weight_shift_norm          # FC weight change magnitude
                )
                score = -raw_score  # NEGATE: Framework expects ID > OOD
            else:
                score = dE  # Fallback
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
                raw_score = (
                    self.w_dE * dE +                                    # Energy change (OOD > ID)
                    self.w_feature * weight_shift_norm * feature_norm +  # Weight shift × feature (OOD > ID paradoxically)
                    weight_feature_alignment * dist_to_prototype_orig    # Alignment × distance (OOD > ID)
                )
                score = -raw_score  # NEGATE: Framework expects ID > OOD
            else:
                score = dE  # Fallback
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
        else:
            # combo - delta_energy, gradnorm, grad_ratio의 가중 합
            score = self.w_dE * dE

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
