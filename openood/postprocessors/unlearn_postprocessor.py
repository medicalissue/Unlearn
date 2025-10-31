from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # 가중치
        wcfg = self.args.get("weights", {}) or {}
        self.w_dE = float(wcfg.get("denergy", 1.0))
        self.w_dC = float(wcfg.get("dconf",   0.5))
        self.w_dM = float(wcfg.get("dmargin", 0.5))
        self.w_G  = float(wcfg.get("g",       0.5))

    # ---------- utils ----------
    @staticmethod
    def _build_fc_from_numpy(w: np.ndarray, b: np.ndarray, device, dtype):
        # w: (C,D) 또는 (D,C), b: (C,)
        C = b.shape[0]
        if w.shape[0] == C:
            # w is (C, D) -> 그대로 사용
            out_features, in_features = w.shape[0], w.shape[1]
            W_torch = torch.from_numpy(w)
        elif w.shape[1] == C:
            # w is (D, C) -> transpose 필요
            out_features, in_features = w.shape[1], w.shape[0]
            W_torch = torch.from_numpy(w.T)
        else:
            raise ValueError(f"Incompatible shapes: w{w.shape}, b{b.shape}")

        fc = nn.Linear(in_features, out_features, bias=True).to(device=device, dtype=dtype)
        with torch.no_grad():
            fc.weight.copy_(W_torch.to(device=device, dtype=dtype))
            fc.bias.copy_(torch.from_numpy(b).to(device=device, dtype=dtype))
        return fc

    @staticmethod
    def _energy(logits: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(logits, dim=-1)

    @staticmethod
    def _margin(logits: torch.Tensor) -> torch.Tensor:
        top2 = torch.topk(logits, k=2, dim=-1).values
        return top2[0] - top2[1]

    @torch.no_grad()
    def _pseudo_target(self, logit_row: torch.Tensor) -> torch.Tensor:
        if self.temp is None or self.temp <= 1.0:
            k = logit_row.argmax(dim=-1)
            return F.one_hot(k, num_classes=self.num_classes).float()
        else:
            return F.softmax(logit_row / self.temp, dim=-1)

    def _per_sample_gradnorm_fc(self, feat_row, target, fc):
        fc.zero_grad(set_to_none=True)
        out = fc(feat_row[None])               # [1, C]
        if target.ndim == 1:                   # one-hot 또는 클래스 분포 [C]
            hard = target.argmax(dim=-1).long().view(1)   # [1], Long
            loss = F.cross_entropy(out, hard)             # CE는 [N,C] vs [N]
        else:
            loss = F.kl_div(F.log_softmax(out, dim=-1), target[None], reduction="batchmean")
        loss.backward()
        g = fc.weight.grad
        return float(torch.sum(torch.abs(g)).detach().cpu().item())

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        # 단일 샘플 입력
        w_np, b_np = net.get_fc()
        logits_orig, feat = net.forward(data, return_feature=True)
        device, dtype = feat.device, feat.dtype

        # 원본 예측
        pred = int(torch.argmax(logits_orig).item())

        # 1) pseudo target 생성
        target = self._pseudo_target(logits_orig).to(device=device, dtype=dtype)

        # 2) FC 복사본 생성
        fc = self._build_fc_from_numpy(w_np, b_np, device, dtype)

        # (선택) GradNorm 계산
        gradnorm_val = None
        if self.use_gradnorm:
            with torch.enable_grad():
                fc.train(False)
                gradnorm_val = self._per_sample_gradnorm_fc(feat, target, fc)

                # 3) 헤드만 ascent
        with torch.enable_grad():
            fc.train(True)
            for _ in range(max(1, self.num_steps)):
                fc.zero_grad(set_to_none=True)
                out = fc(feat[None])                           # [1, C]
                if self.temp is not None and self.temp > 1.0:  # soft target
                    loss = F.kl_div(F.log_softmax(out, dim=-1), target[None], reduction="batchmean")
                else:                                          # hard target
                    hard = target.argmax().long().view(1)      # [1], Long
                    loss = F.cross_entropy(out, hard)
                loss.backward()
                with torch.no_grad():
                    for p in fc.parameters():
                        if p.grad is not None:
                            p.add_(self.eta * p.grad)          # ascent

        # 4) 전/후 스코어 계산
        with torch.no_grad():
            z_before_1d = logits_orig                      # [C]
            z_after_1d  = fc(feat[None]).squeeze(0)        # [C]

            dE = self._energy(z_after_1d) - self._energy(z_before_1d)

            pred = int(torch.argmax(z_before_1d).item())
            p_before = F.softmax(z_before_1d, dim=-1)[pred]
            p_after  = F.softmax(z_after_1d,  dim=-1)[pred]
            dC = (p_before - p_after)

            dM = self._margin(z_before_1d) - self._margin(z_after_1d)

            # score 계산
            if self.score_type in {"delta_energy", "denergy"}:
                score = dE
            elif self.score_type in {"delta_conf", "dconf"}:
                score = dC
            elif self.score_type in {"delta_margin", "dmargin"}:
                score = dM
            elif self.score_type in {"gradnorm", "g"}:
                if gradnorm_val is None:
                    gradnorm_val = self._per_sample_gradnorm_fc(feat, target, fc)
                score = torch.tensor(gradnorm_val, device=device, dtype=dtype)
            else:
                score = (self.w_dE * dE) + (self.w_dC * dC) + (self.w_dM * dM)
                if self.use_gradnorm:
                    if gradnorm_val is None:
                        gradnorm_val = self._per_sample_gradnorm_fc(feat, target, fc)
                    score = score + self.w_G * torch.tensor(gradnorm_val, device=device, dtype=dtype)

        return pred, score
