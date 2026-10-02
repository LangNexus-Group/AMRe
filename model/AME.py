
# =============================任意数量模态融合================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class AME(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.kl = nn.KLDivLoss(reduction="none", log_target=False)
        # self.ce_loss = nn.BCEWithLogitsLoss(reduction="none")  # reduction=none 表示的是忽略掉batch_size维度的
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        # 用于“准确性打分”的指标：precision 或 f1（默认 precision）。
        # 样本级（single-label multiclass）下，precision/F1 的软版本可用 p_true 近似。
        self.acc_metric = getattr(args, 'ame_acc_metric', 'ce_loss')
        self.unc_metric = getattr(args, 'ame_unc_metric', 'kl')
        self.beta = args.ame_beta
        self.gama = args.ame_gama
        self.temperature = args.ame_temperature
        # self.temperature = 1.0
        self.un_target_D = 1/args.num_classes
        self.model_name = eval(args.model_name)


    def _per_sample_acc_score(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute sample-level accuracy score for masking.

        Supported metrics (self.acc_metric):
        - 'precision' (default): 1 - p_true. Simple probability gap.
        - 'margin': 1 - (p_true - p_runner_up). Penalizes confusion with other classes.
        - 'brier': Mean Squared Error between prob distribution and one-hot label.
        - 'ce_loss': Cross Entropy Loss.
        """
        labels = labels.long()
        probs = torch.softmax(logits, dim=-1)
        batch_size = labels.size(0)
        
        # 1. Get probability of the true class
        p_true = probs[torch.arange(batch_size, device=logits.device), labels]

        if self.acc_metric == 'margin':
            # Margin = p_true - max(p_others)
            # Clone probs and mask out true class to find max of remainder
            # print(f"use margin")
            
            probs_clone = probs.clone()
            probs_clone[torch.arange(batch_size, device=logits.device), labels] = -1.0 # eliminate true class
            p_max_other, _ = probs_clone.max(dim=-1)
            
            # margin range is [-1, 1]. 
            # We map it to a loss-like score: 1 - margin.
            # Best case (1.0 margin) -> 0.0 score. Worst case (-1.0 margin) -> 2.0 score.
            margin = p_true - p_max_other
            score = 1.0 - margin


        elif self.acc_metric == 'ce_loss':
            # Cross Entropy Loss
            score = self.ce_loss(logits, labels)
            # print(f"use ce_loss")
            
        else: # default to 'precision' equivalent (1 - p_true)
            score = self.ce_loss(logits, labels)
            
        return score


        
    def forward(self, *modalities, fusion_logits=None, labels=None, epoch_index=-1, epoch=-1):
        """
        接受任意数量的模态作为输入，每个模态都是一个logit张量。
        """
        if not modalities:
            raise ValueError("At least one modality must be provided.")

        num_modalities = len(modalities)
        batch_size = modalities[0].shape[0]
        device = modalities[0].device
        # eps = 1e-8
        
        # ================ 1. 计算准确度分数 ====================
        labels = labels.long()
        # 计算每个模态的交叉熵损失
        acc_scores_list = [self._per_sample_acc_score(m, labels).squeeze() for m in modalities]
        acc_scores = torch.stack(acc_scores_list, dim=-1)  # Shape: [batch, num_modalities]
        eps = 1e-4
        # 生成和 acc_scores 同维度的 eps 张量
        eps_f = torch.full_like(acc_scores, eps)  # 关键代码
        acc_softmax = torch.softmax(acc_scores / self.temperature, dim=-1)  # Shape: [batch, num_modalities]
        
        # ============= 2. 计算不确定性分数 ===============
        
        if self.unc_metric == 'kl':
            # 目标均匀分布
            log_softmax_modalities = [torch.log_softmax(m, dim=-1) for m in modalities]
            target_dis = torch.full_like(log_softmax_modalities[0], self.un_target_D)
            # print(f"target_dis is {target_dis}")
            # 计算每个模态与均匀分布的KL散度作为不确定性
            kl_divs = [-self.kl(log_m, target_dis).sum(dim=-1) for log_m in log_softmax_modalities]
            uncerten_scores = torch.stack(kl_divs, dim=-1)  # Shape: [batch, num_modalities]
        elif self.unc_metric == 'entropy':
            # 使用负信息熵
            # print(f"use entropy")
            prob_modalities = [torch.softmax(m, dim=-1) for m in modalities]
            uncerten_scores = torch.stack(
                [-(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1) for p in prob_modalities],
                dim=-1,
            )  # [B, M]

        uncerten_softmax = torch.softmax(uncerten_scores / self.temperature, dim=-1) # Shape: [batch, num_modalities]

        eps = 1e-8
        final_ratios = acc_softmax * self.beta + uncerten_softmax * (1 - self.beta) # Shape: [batch, num_modalities]
        # # ================= 4. 生成掩码 =================
        # 新逻辑：仅 mask 每个样本中得分最低的模态
        gama = self.gama
        
        # 1) 样本级差异
        max_scores, _ = torch.max(final_ratios, dim=-1)
        min_scores, _ = torch.min(final_ratios, dim=-1)
        diffs = max_scores - min_scores  # [batch]
        
        # 2) 需要掩码的样本（差异 >= gama）
        samples_to_mask = diffs >= gama  # [batch] bool
        # samples_to_mask = (diffs >= gama) & (diffs < 0.9)  # [batch] bool

        # if epoch_index<=5:
        #     diffs_mean = diffs.mean()
        #     diffs_median = diffs.median()
        #     print(f"mean is {diffs_mean.item()} median is {diffs_median.item()}")
        # 3) 生成仅屏蔽最低分模态的掩码
        # 先做全 1，再把最低分索引位置置 0
        min_score_indices = torch.argmin(final_ratios, dim=-1)  # [batch]
        # min_score_indices = torch.argmax(final_ratios, dim=-1)  # [batch]
        one_hot_min = torch.zeros_like(final_ratios)
        one_hot_min.scatter_(dim=1, index=min_score_indices.unsqueeze(1), src=torch.ones_like(min_score_indices, dtype=final_ratios.dtype).unsqueeze(1))
        loser_mask = torch.ones_like(final_ratios) - one_hot_min  # 该掩码仅在最低分位置为 0，其余为 1
        
        # 4) 条件应用掩码：满足条件用 loser_mask，否则全 1
        result = torch.where(samples_to_mask.unsqueeze(1), loser_mask, torch.ones_like(final_ratios))
        # ======================= 5. 日志输出 =====================
        if epoch_index == 0:
            tensors_to_print = {"labels": labels.squeeze()}
            metrics_info = [
                ("Prob", lambda t, i, lbl: t[i, lbl].item()),
                ("Loss", lambda t, i, _: t[i].item()),
                ("AccR", lambda t, i, _: t[i].item()),
                ("UncR", lambda t, i, _: t[i].item()),
                ("FinalR", lambda t, i, _: t[i].item()),
            ]
            # if fusion_logits is not None:
            #     metrics_info.append(("MI", lambda t, i, _: t[i].item()))
            modal_metrics = {}
            for idx, modality in enumerate(modalities):
                mod_name = self.model_name[idx]
                modal_metrics.setdefault(mod_name, {})
                # print(f"modality shape is {modality.shape} for {mod_name}")
                modal_metrics[mod_name]["Prob"] = torch.softmax(modality, dim=1)
                modal_metrics[mod_name]["Loss"] = acc_scores[:, idx]
                modal_metrics[mod_name]["AccR"] = acc_softmax[:, idx]
                modal_metrics[mod_name]["UncR"] = uncerten_softmax[:, idx]
                modal_metrics[mod_name]["FinalR"] = final_ratios[:, idx]
                # if fusion_logits is not None:
                #     modal_metrics[mod_name]["MI"] = mi_modal_pred[:, idx]

            print("\n" + "=" * 200)
            print(f"{'Epoch 0 Detailed Statistics':^200}")
            print("=" * 200)
            header = f"{'Sample':<6} | {'Label':<6} | "
            for metric_name, _ in metrics_info:
                for mod_name in self.model_name:
                    header += f"{mod_name+'_'+metric_name:<12} | "
            header += f"{'Decision':<20}"
            print(header)
            print("-" * (len(header) + 4))

            for i in range(batch_size):
                label_val = int(tensors_to_print['labels'][i].item())
                row = f"{i:<6} | {label_val:<6} | "
                for metric_name, extractor in metrics_info:
                    for mod_name in self.model_name:
                        tensor = modal_metrics[mod_name][metric_name]
                        value = extractor(
                            tensor,
                            i,
                            label_val if metric_name == "Prob" else None
                        )
                        row += f"{value:<12.4f} | "
                masked_modalities = [
                    self.model_name[k] for k, masked in enumerate(result[i]) if masked == 0
                ]
                decision = "None" if not masked_modalities else ", ".join(masked_modalities)
                row += f"{decision:<20}"
                print(row)

            print("\n" + "=" * (len(header) + 4))
        # if 0<epoch_index<=2:
        #     print(f"current_beta is {current_beta}")
        return result.bool(), final_ratios
    
#================Shapley 计算==================

class Shapley(nn.Module):
    def __init__(self, args):
        super(Shapley, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        

    def forward(self, *modalities, fusion_logits=None, labels=None, epoch_index=-1, epoch=-1):

        if not modalities:
            raise ValueError("At least one modality must be provided.")

        num_modalities = len(modalities)
        batch_size = modalities[0].shape[0]
        device = modalities[0].device
        eps = 1e-8
        m1_Mask =torch.ones(batch_size, device=device)
        m2_Mask =torch.ones(batch_size, device=device)
        m3_Mask =torch.ones(batch_size, device=device)

        contribution = {}
         # 计算softmax概率
        # 累计贡献度
        if num_modalities == 2:
            cona, conv = 0.0, 0.0
        elif num_modalities == 3:
            cona, conv, cont = 0.0, 0.0, 0.0

        prediction = self.softmax(fusion_logits)
        modal_preds = [self.softmax(logits) for logits in modalities]
        
        # 计算每个样本的 Shapley 值
        for i, label in enumerate(labels):
            fusion_pred = torch.argmax(prediction[i]).item()
            modal_pred = [torch.argmax(modal_pred[i]).item() for modal_pred in modal_preds]

            # 价值函数：预测正确为1，错误为0
            value_all = 1.0 if fusion_pred == label.item() else 0.0
            modal_values = [1.0 if idx == label.item() else 0.0 for idx in modal_pred]
            value_empty = 0.0  # 空集的价值为0
            
            if num_modalities == 2:
                # 两模态 Shapley 值计算
                # φ_m1 = 1/2 * [v({m1}) - v(∅)] + 1/2 * [v({m1,m2}) - v({m2})]
                # φ_m2 = 1/2 * [v({m2}) - v(∅)] + 1/2 * [v({m1,m2}) - v({m1})]
                
                shapley_m1 = 0.5 * (modal_values[0] - value_empty) + 0.5 * (value_all - modal_values[1])
                shapley_m2 = 0.5 * (modal_values[1] - value_empty) + 0.5 * (value_all - modal_values[0])

                if shapley_m1 < shapley_m2:
                    m1_Mask[i] = 0.0
                elif shapley_m1 > shapley_m2:
                    m2_Mask[i] = 0.0   
            elif num_modalities == 3:

                # 假设我们只有单模态和全模态的输出
                # 简化版 Shapley 值估计：
                shapley_m1 = (modal_values[0] - value_empty) / 3.0 + (value_all - (modal_values[1] + modal_values[2]) / 2.0) / 3.0
                shapley_m2 = (modal_values[1] - value_empty) / 3.0 + (value_all - (modal_values[0] + modal_values[2]) / 2.0) / 3.0
                shapley_m3 = (modal_values[2] - value_empty) / 3.0 + (value_all - (modal_values[0] + modal_values[1]) / 2.0) / 3.0

                # 找出 Shapley 值最大的模态并 mask（让弱模态得到更多训练）
                shapley_values = [shapley_m1, shapley_m2, shapley_m3]
                max_idx = shapley_values.index(max(shapley_values))
                
                if max_idx == 0:
                    m1_Mask[i] = 0.0
                elif max_idx == 1:
                    m2_Mask[i] = 0.0
                else:
                    m3_Mask[i] = 0.0
        
        # 返回 mask，根据模态数量返回对应数量的 mask
        if num_modalities == 2:
            return m1_Mask.bool(), m2_Mask.bool()
        elif num_modalities == 3:
            return m1_Mask.bool(), m2_Mask.bool(), m3_Mask.bool()
        else:
            # 对于其他情况，返回所有 mask
            return m1_Mask, m2_Mask, m3_Mask
