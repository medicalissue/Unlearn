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
        self.w_dC = float(wcfg.get("dconf",   0.5))
        self.w_dM = float(wcfg.get("dmargin", 0.5))
        self.w_G  = float(wcfg.get("g",       0.5))
        # 새로운 메트릭 가중치
        self.w_dH = float(wcfg.get("dentropy", 0.5))
        self.w_dKL = float(wcfg.get("dkl", 0.5))
        self.w_ratio = float(wcfg.get("grad_ratio", 0.5))
        self.w_flip = float(wcfg.get("flip", 0.3))
        self.w_align = float(wcfg.get("alignment", 0.3))
        self.w_consistency = float(wcfg.get("label_consistency", 0.5))

    # ---------- utils ----------

    @staticmethod
    def _energy(logits: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(logits, dim=-1)

    @staticmethod
    def _margin(logits: torch.Tensor) -> torch.Tensor:
        top2 = torch.topk(logits, k=2, dim=-1).values
        return top2[0] - top2[1]

    @staticmethod
    def _entropy(probs: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Calculate entropy: H = -sum(p * log(p))"""
        return -torch.sum(probs * torch.log(probs + eps), dim=-1)

    @staticmethod
    def _kl_from_uniform(probs: torch.Tensor, num_classes: int, eps: float = 1e-10) -> torch.Tensor:
        """Calculate KL divergence from uniform distribution: KL(p || uniform)"""
        uniform = 1.0 / num_classes
        return torch.sum(probs * torch.log((probs + eps) / uniform), dim=-1)

    @staticmethod
    def _prediction_flip(logits_before: torch.Tensor, logits_after: torch.Tensor, k: int = 5) -> torch.Tensor:
        """Calculate prediction flip score based on top-k rank changes"""
        # Get top-k indices for both
        _, idx_before = torch.topk(logits_before, k=min(k, logits_before.shape[-1]), dim=-1)
        _, idx_after = torch.topk(logits_after, k=min(k, logits_after.shape[-1]), dim=-1)

        # Count how many changed
        # Simple version: check if top-1 changed
        flip_score = (idx_before[0] != idx_after[0]).float()

        return flip_score

    @staticmethod
    def _label_consistency(target_init: torch.Tensor, target_current: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Calculate label consistency between initial and current pseudo labels

        Args:
            target_init: Initial pseudo target (one-hot or soft)
            target_current: Current pseudo target (one-hot or soft)
            eps: Small value for numerical stability

        Returns:
            Consistency score: higher means more consistent (ID-like), lower means less consistent (OOD-like)
        """
        # vmap 호환: 조건문 없이 cosine similarity 사용
        # Cosine similarity는 one-hot과 soft label 모두에 잘 작동
        # One-hot: argmax가 같으면 1.0, 다르면 0.0
        # Soft: 분포의 유사도 측정

        target_init_norm = target_init / (torch.norm(target_init, dim=-1, keepdim=True) + eps)
        target_current_norm = target_current / (torch.norm(target_current, dim=-1, keepdim=True) + eps)

        # Cosine similarity
        consistency = torch.sum(target_init_norm * target_current_norm, dim=-1)

        # Normalize to [0, 1] range: (cos + 1) / 2
        # cos=1 (same direction) -> 1.0
        # cos=-1 (opposite) -> 0.0
        consistency = (consistency + 1.0) / 2.0

        return consistency

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

    def _single_sample_unlearn(self, fc_weight, fc_bias, feature, device):
        """단일 샘플에 대한 unlearning 수행 (vmap으로 병렬화될 함수)

        Args:
            fc_weight: FC layer weight [num_classes, feature_dim]
            fc_bias: FC layer bias [num_classes] or None
            feature: 추출된 feature [feature_dim]
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
                    W = W + self.eta * grad_W * (fisher_W + self.fisher_damping)
                else:
                    # Gradient ascent
                    W = W + self.eta * grad_W

        # 업데이트된 FC로 로짓 계산
        with torch.no_grad():
            if b is not None:
                logits_after = F.linear(feature, W, b)
            else:
                logits_after = F.linear(feature, W)

        # 예측
        pred = torch.argmax(logits_orig, dim=-1, keepdim=True)

        # 스코어 계산 (NaN 방지를 위해 logit 정규화)
        # Max를 빼서 상대적 크기는 유지하면서 수치적 안정성 확보
        logits_orig_max = logits_orig.max(dim=-1, keepdim=True).values
        logits_after_max = logits_after.max(dim=-1, keepdim=True).values

        logits_orig_stable = logits_orig - logits_orig_max
        logits_after_stable = logits_after - logits_after_max

        dE = self._energy(logits_after_stable) - self._energy(logits_orig_stable)

        # vmap 호환: .item() 대신 gather 사용
        pred_idx = torch.argmax(logits_orig_stable, dim=-1, keepdim=True)
        p_before = torch.gather(F.softmax(logits_orig_stable, dim=-1), -1, pred_idx).squeeze(-1)
        p_after = torch.gather(F.softmax(logits_after_stable, dim=-1), -1, pred_idx).squeeze(-1)
        dC = p_before - p_after

        dM = self._margin(logits_orig_stable) - self._margin(logits_after_stable)

        # 새로운 메트릭 계산
        # Entropy change
        probs_before = F.softmax(logits_orig_stable, dim=-1)
        probs_after = F.softmax(logits_after_stable, dim=-1)
        H_before = self._entropy(probs_before)
        H_after = self._entropy(probs_after)
        dH = H_after - H_before  # OOD는 엔트로피 증가

        # KL divergence from uniform
        KL_before = self._kl_from_uniform(probs_before, self.num_classes)
        KL_after = self._kl_from_uniform(probs_after, self.num_classes)
        dKL = KL_before - KL_after  # OOD는 균등 분포에 가까워짐

        # Prediction flip
        flip = self._prediction_flip(logits_orig_stable, logits_after_stable)

        # Feature alignment (weight change와 feature의 내적)
        delta_W = W - fc_weight
        # Flatten for dot product
        delta_W_flat = delta_W.reshape(-1)
        feature_norm = torch.norm(feature) + 1e-10
        delta_W_norm = torch.norm(delta_W_flat) + 1e-10
        # feature와 각 클래스 weight 변화의 평균 alignment
        alignment = torch.dot(feature, delta_W.t().reshape(-1)[:feature.shape[0]]) / (feature_norm * delta_W_norm)

        # Label consistency (초기 target과 최종 target의 일관성)
        # OOD는 낮은 consistency, ID는 높은 consistency
        label_consistency = self._label_consistency(target_init, target_final)
        # OOD 스코어로 변환: 낮은 consistency = 높은 OOD 스코어!@@@@@@@@@@@@@@@@@
        inconsistency = label_consistency

        # 스코어 타입에 따라 계산
        if self.score_type in {"delta_energy", "denergy"}:
            score = -dE
        elif self.score_type in {"delta_conf", "dconf"}:
            score = -dC
        elif self.score_type in {"delta_margin", "dmargin"}:
            score = -dM
        elif self.score_type in {"delta_entropy", "dentropy"}:
            score = dH
        elif self.score_type in {"delta_kl", "dkl"}:
            score = -dKL
        elif self.score_type == "flip":
            score = flip
        elif self.score_type == "alignment":
            score = alignment
        elif self.score_type in {"label_consistency", "consistency"}:
            score = inconsistency  # 낮은 consistency = 높은 OOD 스코어
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

            score = gradnorm_o - gradnorm_val if gradnorm_o is not None else -gradnorm_val
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
                score = 1.0 - ratio  # OOD는 gradient가 많이 줄어듦
            else:
                score = -gradnorm_after  # gradnorm_o가 없으면 after만 사용
        else:
            # combo - 모든 메트릭의 가중 합
            score = (self.w_dE * dE) - (self.w_dC * dC) - (self.w_dM * dM)

            # 새로운 메트릭 추가
            score = score + (self.w_dH * dH) + (self.w_dKL * dKL) + (self.w_flip * flip) + (self.w_align * alignment)
            score = score + (self.w_consistency * inconsistency)  # label consistency 추가

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

                # Gradnorm difference
                gradnorm_contrib = gradnorm_o - gradnorm_after
                score = score + (self.w_G * gradnorm_contrib)

                # Gradient ratio (vmap 호환)
                ratio = gradnorm_after / (gradnorm_o + 1e-10)
                ratio_contrib = 1.0 - ratio
                score = score + (self.w_ratio * ratio_contrib)

        # NaN이나 Inf가 있으면 0으로 대체
        score = torch.where(torch.isnan(score) | torch.isinf(score),
                           torch.tensor(0.0, device=device, dtype=score.dtype),
                           score)

        return pred, score

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

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

        # vmap을 사용하여 배치 병렬화
        if fc_biases is not None:
            vmapped_fn = vmap(
                lambda w, b, f: self._single_sample_unlearn(w, b, f, device),
                in_dims=(0, 0, 0),  # fc_weight, fc_bias, feature
                out_dims=(0, 0)  # pred, score
            )
            preds, scores = vmapped_fn(fc_weights, fc_biases, features)
        else:
            vmapped_fn = vmap(
                lambda w, f: self._single_sample_unlearn(w, None, f, device),
                in_dims=(0, 0),  # fc_weight, feature
                out_dims=(0, 0)  # pred, score
            )
            preds, scores = vmapped_fn(fc_weights, features)

        return preds.squeeze(-1), scores
