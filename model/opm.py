import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Tuple


class ModalityDropper(nn.Module):
    """Implements OPM-style feed-forward modality dropping."""

    def __init__(
        self,
        num_modalities: int,
        execute_prob: float = 0.7,
        q_base: float = 0.5,
        lam: float = 0.5,
        min_keep: int = 1,
        max_drop: float = 0.95,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.execute_prob = execute_prob
        self.q_base = q_base
        self.lam = lam
        self.min_keep = max(0, min_keep)
        self.max_drop = max_drop
        self.keep_floor = 0.0
        self.use_bml_formula = num_modalities == 2
        self.cpu_generator = torch.Generator()
        self.cpu_generator.manual_seed(seed)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        logits: Sequence[torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], Optional[Dict[str, object]]]:
        if not self.training:
            return list(features), None
        if labels is None or len(features) < 2 or len(logits) < len(features):
            return list(features), None
        if self.execute_prob <= 0:
            return list(features), {'executed': False}
        batch_size = labels.size(0)
        if batch_size == 0:
            return list(features), None
        if torch.rand(1, generator=self.cpu_generator).item() > self.execute_prob:
            return list(features), {'executed': False}

        target = labels.view(-1, 1)
        score_mean_list = []
        score_sum_list = []
        for logit in logits:
            probs = F.softmax(logit.detach(), dim=-1)
            gathered = probs.gather(1, target)
            score_mean_list.append(gathered.mean())
            score_sum_list.append(gathered.sum())
        mean_scores = torch.stack(score_mean_list)
        score_sums = torch.stack(score_sum_list)

        if self.use_bml_formula:
            drop_probs = self._calc_bml_drop(score_sums)
        else:
            mean_score = mean_scores.mean().clamp_min(1e-6)
            discrepancy = (mean_scores / mean_score) - 1.0
            drop_probs = torch.clamp(
                self.q_base + self.lam * discrepancy,
                min=0.0,
                max=self.max_drop,
            )
        keep_probs = (1.0 - drop_probs).clamp(self.keep_floor, 1.0)

        keep_probs_cpu = keep_probs.detach().cpu()
        samples = torch.rand(
            (batch_size, self.num_modalities),
            generator=self.cpu_generator,
        )
        keep_mask = (samples <= keep_probs_cpu.unsqueeze(0)).float()
        min_keep = min(self.min_keep, self.num_modalities)
        if min_keep > 0:
            needs_fix = torch.nonzero(keep_mask.sum(dim=1) < min_keep, as_tuple=False).flatten()
            if needs_fix.numel() > 0:
                top_idx = torch.topk(keep_probs, k=min_keep).indices
                for row in needs_fix:
                    keep_mask[row, top_idx] = 1.0

        keep_mask = keep_mask.to(features[0].device)
        keep_view = [
            keep_mask[:, idx].view(-1, *([1] * (feat.dim() - 1)))
            for idx, feat in enumerate(features)
        ]

        scaled_features: List[torch.Tensor] = []
        if self.use_bml_formula:
            dim_tensor = torch.tensor(
                [feat.shape[1] if feat.dim() > 1 else 1 for feat in features],
                dtype=drop_probs.dtype,
                device=drop_probs.device,
            )
            theta = torch.dot(dim_tensor, drop_probs) / dim_tensor.sum().clamp_min(1e-6)
            theta = theta.clamp(max=0.999)
            scale = 1.0 / (1.0 - theta)
            for idx, feat in enumerate(features):
                scaled = feat * keep_view[idx] * scale
                scaled_features.append(scaled)
        else:
            for idx, feat in enumerate(features):
                scale = 1.0 / keep_probs[idx].clamp_min(1e-3)
                scaled = feat * keep_view[idx] * scale
                scaled_features.append(scaled)

        keep_ratio = keep_mask.mean(dim=0)
        drop_counts = (1.0 - keep_mask).sum(dim=0)
        drop_info = {
            'executed': True,
            'keep_ratio': keep_ratio.detach().cpu().tolist(),
            'drop_probs': drop_probs.detach().cpu().tolist(),
            'scores': mean_scores.detach().cpu().tolist(),
            'score_sums': score_sums.detach().cpu().tolist(),
            'drop_counts': drop_counts.detach().cpu().tolist(),
            'batch_size': batch_size,
        }
        return scaled_features, drop_info

    def _calc_bml_drop(self, perf_sums: torch.Tensor) -> torch.Tensor:
        if perf_sums.numel() != 2:
            return torch.zeros_like(perf_sums)
        eps = 1e-6
        perf1, perf2 = perf_sums[0].clamp_min(eps), perf_sums[1].clamp_min(eps)
        ratio1 = torch.tanh(torch.relu(perf1 / perf2 - 1.0))
        ratio2 = torch.tanh(torch.relu(perf2 / perf1 - 1.0))
        ratio1_val = float(ratio1.detach().cpu())
        ratio2_val = float(ratio2.detach().cpu())
        q = torch.zeros_like(perf_sums)
        if ratio1_val > 0:
            q_val = self.q_base * (1.0 + self.lam * ratio1_val)
            q[0] = min(max(q_val, 0.0), self.max_drop)
        if ratio2_val > 0:
            q_val = self.q_base * (1.0 + self.lam * ratio2_val)
            q[1] = min(max(q_val, 0.0), self.max_drop)
        return q

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict, List, Optional, Sequence, Tuple


# class ModalityDropper(nn.Module):
#     """Implements OPM-style feed-forward modality dropping."""

#     def __init__(
#         self,
#         num_modalities: int,
#         execute_prob: float = 0.7,
#         q_base: float = 0.5,
#         lam: float = 0.5,
#         min_keep: int = 1,
#         max_drop: float = 0.95,
#         seed: int = 0,
#     ) -> None:
#         super().__init__()
#         self.num_modalities = num_modalities
#         self.execute_prob = execute_prob
#         self.q_base = q_base
#         self.lam = lam
#         self.min_keep = max(0, min_keep)
#         self.max_drop = max_drop
#         self.keep_floor = 0.05
#         self.cpu_generator = torch.Generator()
#         self.cpu_generator.manual_seed(seed)

#     def forward(
#         self,
#         features: Sequence[torch.Tensor],
#         logits: Sequence[torch.Tensor],
#         labels: torch.Tensor,
#     ) -> Tuple[List[torch.Tensor], Optional[Dict[str, object]]]:
#         if not self.training:
#             return list(features), None
#         if labels is None or len(features) < 2 or len(logits) < len(features):
#             return list(features), None
#         if self.execute_prob <= 0:
#             return list(features), {'executed': False}
#         batch_size = labels.size(0)
#         if batch_size == 0:
#             return list(features), None
#         if torch.rand(1, generator=self.cpu_generator).item() > self.execute_prob:
#             return list(features), {'executed': False}

#         target = labels.view(-1, 1)
#         score_list = []
#         for logit in logits:
#             probs = F.softmax(logit.detach(), dim=-1)
#             gathered = probs.gather(1, target)
#             score_list.append(gathered.mean())
#         scores = torch.stack(score_list)
#         mean_score = scores.mean().clamp_min(1e-6)
#         discrepancy = (scores / mean_score) - 1.0
#         drop_probs = torch.clamp(
#             self.q_base + self.lam * discrepancy,
#             min=0.0,
#             max=self.max_drop,
#         )
#         keep_probs = (1.0 - drop_probs).clamp(self.keep_floor, 1.0)

#         keep_probs_cpu = keep_probs.detach().cpu()
#         samples = torch.rand(
#             (batch_size, self.num_modalities),
#             generator=self.cpu_generator,
#         )
#         keep_mask = (samples <= keep_probs_cpu.unsqueeze(0)).float()
#         min_keep = min(self.min_keep, self.num_modalities)
#         if min_keep > 0:
#             needs_fix = torch.nonzero(keep_mask.sum(dim=1) < min_keep, as_tuple=False).flatten()
#             if needs_fix.numel() > 0:
#                 top_idx = torch.topk(keep_probs, k=min_keep).indices
#                 for row in needs_fix:
#                     keep_mask[row, top_idx] = 1.0

#         keep_mask = keep_mask.to(features[0].device)
#         keep_view = [
#             keep_mask[:, idx].view(-1, *([1] * (feat.dim() - 1)))
#             for idx, feat in enumerate(features)
#         ]

#         scaled_features: List[torch.Tensor] = []
#         for idx, feat in enumerate(features):
#             scale = 1.0 / keep_probs[idx].clamp_min(1e-3)
#             scaled = feat * keep_view[idx] * scale
#             scaled_features.append(scaled)

#         keep_ratio = keep_mask.mean(dim=0)
#         drop_counts = (1.0 - keep_mask).sum(dim=0)
#         drop_info = {
#             'executed': True,
#             'keep_ratio': keep_ratio.detach().cpu().tolist(),
#             'drop_probs': drop_probs.detach().cpu().tolist(),
#             'scores': scores.detach().cpu().tolist(),
#             'drop_counts': drop_counts.detach().cpu().tolist(),
#             'batch_size': batch_size,
#         }
#         return scaled_features, drop_info
