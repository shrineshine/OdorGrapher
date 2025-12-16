import torch
import torch.nn.functional as F
from typing import Optional, Sequence
from deepchem.models.losses import Loss

class AdaptiveLogitAdjustedASL(Loss):
    def __init__(self,
                 class_imbalance_ratio: Sequence[float],
                 tau: float = 0.5,
                 gamma_pos: float = 0.0,
                 gamma_neg_head: float = 4.0,
                 gamma_neg_tail: float = 2.5,
                 ir_threshold: float = 3.0,
                 clip: float = 0.05,
                 beta_pos: float = 0.5,
                 reduction: str = "sum",
                 gap_weight: float = 0.1,
                 gap_margin: float = 0.3,
                 gap_temp_neg: float = 0.5):
        super().__init__()
        assert reduction in ("sum", "mean")
        self.reduction = reduction
        self.tau = tau
        self.gamma_pos = gamma_pos
        self.gamma_neg_head = gamma_neg_head
        self.gamma_neg_tail = gamma_neg_tail
        self.ir_threshold = ir_threshold
        self.clip = clip
        self.beta_pos = beta_pos
        self.gap_weight = gap_weight
        self.gap_margin = gap_margin
        self.gap_temp_neg = gap_temp_neg

        r = torch.as_tensor(class_imbalance_ratio, dtype=torch.float32).clamp(min=1e-6)
        self.r_cpu = r
        pi = (1.0 / (1.0 + r)).clamp(min=1e-6, max=1.0 - 1e-6)
        self.logit_bias_cpu = self.tau * torch.log(pi / (1.0 - pi))

        mean_r = r.mean()
        self.rare_mask_cpu = (r > float(self.ir_threshold))
        self.pos_boost_cpu = (r / mean_r.clamp_min(1e-12)).pow(self.beta_pos)

        self.gamma_neg_vec_cpu = torch.where(
            self.rare_mask_cpu,
            torch.full_like(r, float(self.gamma_neg_tail)),
            torch.full_like(r, float(self.gamma_neg_head)),
        )

        self._cache = {}

    def _to_device_cached(self, ref: torch.Tensor, t_cpu: Optional[torch.Tensor], keep_dtype: bool = False):
        if t_cpu is None:
            return None
        key = (ref.device, ref.dtype if not keep_dtype else t_cpu.dtype, id(t_cpu))
        if key not in self._cache:
            self._cache[key] = t_cpu.to(device=ref.device, dtype=(t_cpu.dtype if keep_dtype else ref.dtype))
        return self._cache[key]

    def _create_pytorch_loss(self):
        eps = 1e-8
        clip = self.clip

        def main_la_asl(logits, labels, bias, pos_boost, gamma_neg_vec):
            labels_f = labels.float()
            logits_adj = logits + bias
            p = torch.sigmoid(logits_adj)

            p_neg = (p - clip).clamp(min=0.0) if clip > 0 else p
            gpos = self.gamma_pos
            gneg = gamma_neg_vec.unsqueeze(0)

            pos_loss = -labels_f * ((1 - p).clamp(min=eps) ** gpos) * torch.log(p.clamp(min=eps))
            pos_loss = pos_loss * pos_boost.unsqueeze(0)

            neg_loss = -(1 - labels_f) * (p_neg.clamp(min=eps) ** gneg) * torch.log((1 - p).clamp(min=eps))
            loss_mat = pos_loss + neg_loss

            return loss_mat.sum(dim=1).mean() if self.reduction == "sum" else loss_mat.mean(dim=1).mean()

        def soft_hard_gap(logits, labels, rare_mask):
            if self.gap_weight <= 0:
                return logits.new_zeros(())

            B, T = logits.shape
            y = labels.float()

            pos_mask_col = (y > 0.5).sum(dim=0) > 0
            neg_mask_col = (y <= 0.5).sum(dim=0) > 0
            valid = pos_mask_col & neg_mask_col & rare_mask

            if valid.sum() == 0:
                return logits.new_zeros(())

            z = logits[:, valid]
            yv = y[:, valid]
            pos_mask = (yv > 0.5)
            neg_mask = ~pos_mask

            pos_cnt = pos_mask.sum(dim=0)
            pos_mean = torch.where(
                pos_cnt > 0,
                (z * pos_mask.float()).sum(dim=0) / pos_cnt.clamp_min(1.0),
                torch.zeros_like(pos_cnt, dtype=z.dtype)
            )

            temp = max(1e-3, float(self.gap_temp_neg))
            minus_inf = torch.finfo(z.dtype).min
            z_masked = torch.where(neg_mask, z / temp, torch.full_like(z, minus_inf))

            w = torch.softmax(z_masked, dim=0)
            w = torch.where(neg_mask, w, torch.zeros_like(w))
            w = w / w.sum(dim=0, keepdim=True).clamp_min(eps)
            neg_soft = (w * z).sum(dim=0)

            gap = F.relu(self.gap_margin - (pos_mean - neg_soft))
            return gap.mean()

        def loss(logits, labels):
            logits_ = logits.squeeze(-1) if logits.dim() == 3 and logits.size(-1) == 1 else logits
            labels_ = labels.squeeze(-1) if labels.dim() == 3 and labels.size(-1) == 1 else labels

            bias = self._to_device_cached(logits_, self.logit_bias_cpu)
            pos_boost = self._to_device_cached(logits_, self.pos_boost_cpu)
            gamma_nv = self._to_device_cached(logits_, self.gamma_neg_vec_cpu)
            rare_mask = self._to_device_cached(logits_, self.rare_mask_cpu, keep_dtype=True)

            main = main_la_asl(logits_, labels_, bias, pos_boost, gamma_nv)
            gap = soft_hard_gap(logits_, labels_, rare_mask)

            return main + self.gap_weight * gap

        return loss