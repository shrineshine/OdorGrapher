import math
from typing import Sequence

import torch
import torch.nn.functional as F
from deepchem.models.losses import Loss


class LogitAdjASLTopPush(Loss):
    """
    Simplified tail-oriented loss after ablation:
    - keep logit adjustment,
    - keep asymmetric focal treatment on negatives,
    - keep hardest-positive top-push ranking,
    - remove tail positive boost,
    - remove hard negative scaling.
    """

    def __init__(self,
                 class_imbalance_ratio: Sequence[float],
                 tau: float = 0.2,
                 gamma_pos: float = 0.0,
                 gamma_neg_head: float = 4.0,
                 gamma_neg_tail: float = 2.0,
                 ir_threshold: float = 3.0,
                 clip: float = 0.05,
                 reduction: str = "sum",
                 toppush_weight: float = 0.035,
                 toppush_margin: float = 0.32,
                 toppush_temp: float = 0.65,
                 toppush_tail_scale: float = 0.85,
                 toppush_neg_topk: int = 5):
        super().__init__()
        assert reduction in ("sum", "mean")
        self.reduction = reduction

        self.tau = float(tau)
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg_head = float(gamma_neg_head)
        self.gamma_neg_tail = float(gamma_neg_tail)
        self.ir_threshold = float(ir_threshold)
        self.clip = float(clip)

        self.toppush_weight = float(toppush_weight)
        self.toppush_margin = float(toppush_margin)
        self.toppush_temp = float(toppush_temp)
        self.toppush_tail_scale = float(toppush_tail_scale)
        self.toppush_neg_topk = int(toppush_neg_topk)

        r = torch.as_tensor(class_imbalance_ratio, dtype=torch.float32).clamp(min=1e-6)
        self.r_cpu = r

        pi = 1.0 / (1.0 + r)
        pi = pi.clamp(min=1e-6, max=1.0 - 1e-6)
        self.logit_bias_cpu = self.tau * torch.log(pi / (1.0 - pi))

        log_r = torch.log(r)
        log_threshold = torch.log(torch.tensor(float(self.ir_threshold), dtype=torch.float32))
        smooth_tail = torch.sigmoid(2.0 * (log_r - log_threshold))
        self.tail_weight_cpu = smooth_tail

        self.gamma_neg_vec_cpu = (
            smooth_tail * float(self.gamma_neg_tail)
            + (1.0 - smooth_tail) * float(self.gamma_neg_head)
        )

        self._cache = {}

    def _to_device_cached(self,
                          ref: torch.Tensor,
                          t_cpu: torch.Tensor,
                          keep_dtype: bool = False) -> torch.Tensor:
        key = (ref.device, t_cpu.dtype if keep_dtype else ref.dtype, id(t_cpu))
        if key not in self._cache:
            if keep_dtype:
                self._cache[key] = t_cpu.to(device=ref.device)
            else:
                self._cache[key] = t_cpu.to(device=ref.device, dtype=ref.dtype)
        return self._cache[key]

    def _create_pytorch_loss(self):
        eps = 1e-8
        clip = self.clip

        def main_loss(logits_: torch.Tensor,
                      labels_: torch.Tensor,
                      bias: torch.Tensor,
                      gamma_neg_vec: torch.Tensor) -> torch.Tensor:
            y = labels_.float()
            logits_adj = logits_ + bias.unsqueeze(0)

            p_pos = torch.sigmoid(logits_adj)
            p_neg_raw = torch.sigmoid(logits_adj)
            p_neg = (p_neg_raw - clip).clamp(min=0.0) if clip > 0 else p_neg_raw

            gpos = self.gamma_pos
            gneg = gamma_neg_vec.unsqueeze(0)

            pos_loss = -y * ((1.0 - p_pos).clamp(min=eps) ** gpos) * torch.log(p_pos.clamp(min=eps))
            neg_loss = -(1.0 - y) * (p_neg.clamp(min=eps) ** gneg) * torch.log((1.0 - p_neg_raw).clamp(min=eps))

            loss_mat = pos_loss + neg_loss
            if self.reduction == "sum":
                return loss_mat.sum(dim=1).mean()
            return loss_mat.mean(dim=1).mean()

        def hardest_positive_toppush(logits_: torch.Tensor,
                                     labels_: torch.Tensor,
                                     bias: torch.Tensor,
                                     tail_weight: torch.Tensor) -> torch.Tensor:
            if self.toppush_weight <= 0:
                return logits_.new_zeros(())

            y = labels_.float()
            active = (y.sum(dim=1) >= 2.0)
            if active.sum() == 0:
                return logits_.new_zeros(())

            z = logits_[active] + bias.unsqueeze(0)
            y = y[active]
            pos_mask = y > 0.5
            neg_mask = ~pos_mask

            rarity = 1.0 + self.toppush_tail_scale * tail_weight

            inf = torch.full_like(z, float("inf"))
            pos_scores = torch.where(pos_mask, z, inf)
            hard_pos_vals, hard_pos_idx = pos_scores.min(dim=1)

            neg_scores = torch.where(neg_mask, z, torch.full_like(z, -1e9))
            kneg = min(max(1, self.toppush_neg_topk), neg_scores.size(1) - 1)
            topk_neg = torch.topk(neg_scores, k=kneg, dim=1).values
            temp = max(1e-3, self.toppush_temp)
            hard_neg = temp * (
                torch.logsumexp(topk_neg / temp, dim=1) - math.log(float(kneg))
            )

            hard_pos_rarity = rarity[hard_pos_idx]
            sample_margin = self.toppush_margin * (1.0 + 0.10 * (hard_pos_rarity - 1.0))
            push = F.relu(sample_margin - (hard_pos_vals - hard_neg))
            return (push * hard_pos_rarity).mean()

        def loss(logits: torch.Tensor, labels: torch.Tensor):
            logits_ = logits.squeeze(-1) if (logits.dim() == 3 and logits.size(-1) == 1) else logits
            labels_ = labels.squeeze(-1) if (labels.dim() == 3 and labels.size(-1) == 1) else labels

            logits_ = logits_.contiguous()
            labels_ = labels_.contiguous()

            bias = self._to_device_cached(logits_, self.logit_bias_cpu)
            gamma_neg_vec = self._to_device_cached(logits_, self.gamma_neg_vec_cpu)
            tail_weight = self._to_device_cached(logits_, self.tail_weight_cpu)

            main = main_loss(logits_, labels_, bias, gamma_neg_vec)
            rank = hardest_positive_toppush(logits_, labels_, bias, tail_weight)
            return main + self.toppush_weight * rank

        return loss
