# import os
# import sys
# # 获取当前文件所在目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 添加到sys.path
# sys.path.insert(0, current_dir)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from collections import defaultdict
from transformers import BertModel, BertTokenizer
from timm.models.vision_transformer import Block
from positional_encodings.torch_encodings import PositionalEncoding1D

from .AME import AME, Shapley
from .utils import CoAttentionExpert
from .AME_Attention_Fusion import AME_AttentionFusion
from .AME_DynamicBetaSoftGate import AME_DynamicBetaSoftGate
from .opm import ModalityDropper



def _sample_random_modality_masks(
    batch_size: int,
    num_modalities: int,
    drop_prob: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator=None,
) -> torch.Tensor:
    """Sample per-sample modality masks with random dropping.

    Returns a (B, M) tensor in {0,1} (float dtype) where 1 means keep.
    Guarantee: each sample keeps at least one modality.
    """
    if num_modalities <= 0:
        raise ValueError(f"num_modalities must be positive, got {num_modalities}")

    # Fast path: keep all.
    if drop_prob <= 0.0:
        return torch.ones(batch_size, num_modalities, device=device, dtype=dtype)

    # If drop_prob>=1, keep exactly one random modality per sample.
    if drop_prob >= 1.0:
        if generator is None:
            keep = torch.zeros(batch_size, num_modalities, device=device, dtype=dtype)
            idx = torch.randint(0, num_modalities, (batch_size,), device=device)
            keep[torch.arange(batch_size, device=device), idx] = 1.0
            return keep
        keep_cpu = torch.zeros(batch_size, num_modalities, device='cpu', dtype=dtype)
        idx_cpu = torch.randint(0, num_modalities, (batch_size,), device='cpu', generator=generator)
        keep_cpu[torch.arange(batch_size, device='cpu'), idx_cpu] = 1.0
        return keep_cpu.to(device=device, dtype=dtype)

    if generator is None:
        keep = (torch.rand(batch_size, num_modalities, device=device) > drop_prob).to(dtype)
        all_dropped = keep.sum(dim=1) == 0
        if all_dropped.any():
            rows = all_dropped.nonzero(as_tuple=False).squeeze(1)
            idx = torch.randint(0, num_modalities, (rows.numel(),), device=device)
            keep[rows] = 0
            keep[rows, idx] = 1.0
        return keep

    # Reproducible path: sample on CPU with an isolated generator, then move to target device.
    keep_cpu = (torch.rand(batch_size, num_modalities, device='cpu', generator=generator) > drop_prob).to(dtype)
    all_dropped_cpu = keep_cpu.sum(dim=1) == 0
    if all_dropped_cpu.any():
        rows_cpu = all_dropped_cpu.nonzero(as_tuple=False).squeeze(1)
        idx_cpu = torch.randint(0, num_modalities, (rows_cpu.numel(),), device='cpu', generator=generator)
        keep_cpu[rows_cpu] = 0
        keep_cpu[rows_cpu, idx_cpu] = 1.0
    return keep_cpu.to(device=device, dtype=dtype)



class base_model(nn.Module):
    def __init__(self, args):
        super(base_model, self).__init__()
        self.batch_size = args.batch_size
        self.unified_dim = args.unified_dim
        self.outdim = args.num_classes
        self.args = args
        self.model_name = eval(args.model_name)
        self.use_OLA = getattr(args, 'Use_OLA', False)
        # Priority: args.random_mask_seed > args.random_seed > 0
        self._random_mask_gen = torch.Generator(device='cpu')
        seed = int(getattr(args, 'random_mask_seed', getattr(args, 'random_seed', 2026)))
        self._random_mask_gen.manual_seed(seed)

        # OPM
        self.model_name = eval(args.model_name)
        self.latest_drop_info = None
        self.opm_controller = None
        if getattr(args, 'use_opm', False) and len(self.model_name) >= 2:
            self.opm_controller = ModalityDropper(
                num_modalities=len(self.model_name),
                execute_prob=getattr(args, 'opm_execute_prob', 0.7),
                q_base=getattr(args, 'opm_q_base', 0.5),
                lam=getattr(args, 'opm_lambda', 0.5),
                min_keep=getattr(args, 'opm_min_keep', 1),
                seed=getattr(args, 'opm_seed', 2024),
            )
        
        # 注册Mask Number 的buffer
        self.register_buffer('m1_mask_num', torch.tensor(0.0)) # 代表视觉类
        self.register_buffer('m2_mask_num', torch.tensor(0.0)) # 代表文本类
        self.register_buffer('m3_mask_num', torch.tensor(0.0))

        self.gap = args.ame_gap
        # self.gap_start = args.ame_gap_start
        
        # 初始化AME模块
        self.MaskType = args.MaskType
        if self.MaskType == "None":
            self.Mask = None
        elif self.MaskType == "Random":
            # Random modality dropping; no extra module required.
            self.Mask = None
        elif self.MaskType == "AME":
            self.Mask = AME(args=args)
        elif self.MaskType == "AME_attention":
            self.Mask = AME_AttentionFusion(args=args)
        elif self.MaskType == "AME_DynamicBetaSoftGate":
            self.Mask = AME_DynamicBetaSoftGate(args=args)
        elif self.MaskType == "Shapley":
            self.Mask = Shapley(args=args)
        else:
            self.Mask = None
        # self.m1_classifier = nn.Sequential(
        #     nn.Linear(self.unified_dim, self.outdim),
        #     nn.SiLU(),
            
        # )
        # self.m2_classifier = nn.Sequential(
        #     nn.Linear(self.unified_dim, 64),
        #     nn.SiLU(),
            
        # )
        self.m1_classifier = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )
        self.m2_classifier = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )
        # self.m1_classifier = nn.Sequential(
        #     nn.Linear(self.unified_dim, self.outdim)
        # )
        # self.m2_classifier = nn.Sequential(
        #     nn.Linear(self.unified_dim, self.outdim)
        # )
        # 存储Mask的状态
        self.Mask_Dict_m1 = defaultdict(lambda: torch.tensor(1, device='cuda:0'))
        self.Mask_Dict_m2 = defaultdict(lambda: torch.tensor(1, device='cuda:0'))
        # 缓存特征用于可视化（不改变forward返回值）
        self.cached_fusion_feature = None
        self.cached_m1_feature = None
        self.cached_m2_feature = None
        
    
    def AME_MASK(self, epoch, batch_size, m1_feature, m2_feature, device, labels=None, epoch_index=-1, fusion_logits=None, sid=None, m1_only_output=None, m2_only_output=None):
        # OPM
        drop_info = None
        if (
            self.opm_controller is not None
            and self.training
            and labels is not None
            and epoch > self.args.warmup_epoch
        ):
            raw_features = [m1_feature, m2_feature]
            with torch.no_grad():
                logits_for_drop = [
                    self.m1_classifier(raw_features[0].detach()),
                    self.m2_classifier(raw_features[1].detach())
                ]
            dropped_features, drop_info = self.opm_controller(raw_features, logits_for_drop, labels)
            m1_feature, m2_feature = dropped_features
        self.latest_drop_info = drop_info
        
        m1_Mask =torch.ones(batch_size, device=device)
        m2_Mask =torch.ones(batch_size, device=device)
        # 生成Mask
        # 训练弱模态
        # if epoch > self.args.warmup_epoch and self.Mask is not None and epoch%3==2:
        # if epoch > self.args.warmup_epoch and self.Mask is not None and epoch%self.gap != self.gap_start:
        # if epoch > self.args.warmup_epoch and self.Mask is not None and epoch%self.gap != 1:
        if epoch > self.args.warmup_epoch and self.MaskType != "None" and epoch % self.gap != 1:
        # if epoch > self.args.warmup_epoch and self.Mask is not None:
            if self.MaskType == 'Random':
                drop_prob = float(getattr(self.args, 'random_mask_drop_prob', 0.0))
                masks = _sample_random_modality_masks(
                    batch_size=batch_size,
                    num_modalities=2,
                    drop_prob=drop_prob,
                    device=device,
                    dtype=m1_feature.dtype,
                    generator=self._random_mask_gen,
                )
                m1_Mask = masks[:, 0]
                m2_Mask = masks[:, 1]
            # else:
            #     with torch.no_grad():
            #         m1_only_output = self.m1_classifier(m1_feature)
            #         m2_only_output = self.m2_classifier(m2_feature)
                    # fusion_logit = self.
            if m1_only_output is None:
                with torch.no_grad():
                    m1_only_output = self.m1_classifier(m1_feature)
                    m2_only_output = self.m2_classifier(m2_feature)
            if self.MaskType == 'AME':
                kl_result,IT_gsd = self.Mask(
                    m1_only_output,
                    m2_only_output,
                    fusion_logits=fusion_logits,
                    labels=labels,
                    epoch_index=epoch_index,
                    epoch=epoch
                )
                # 计算掩码值
                m1_Mask = kl_result[:,0]
                m2_Mask = kl_result[:,1]

            elif self.MaskType == 'AME_attention':
                weights, stats = self.Mask(
                    m1_only_output,
                    m2_only_output,
                    labels=labels,
                    epoch_index=epoch_index,
                    epoch=epoch
                )
                m1_Mask = weights[:,0]
                m2_Mask = weights[:,1]
            
            elif self.MaskType == 'Shapley':
                m1_Mask, m2_Mask = self.Mask(
                    m1_only_output,
                    m2_only_output,
                    fusion_logits=fusion_logits,
                    labels=labels,
                    epoch_index=epoch_index,
                    epoch=epoch,
                    
                )
            if m1_Mask.dim() == 2:
                m1_Mask = m1_Mask.squeeze(1)
                m2_Mask = m2_Mask.squeeze(1)
       
    
            # self.m1_mask_num += (batch_size - m1_Mask.sum().item())
            # self.m2_mask_num += (batch_size - m2_Mask.sum().item())
    
            # m1_feature_masked = m1_Mask.unsqueeze(1).expand_as(m1_feature).bool()
            # m2_feature_masked = m2_Mask.unsqueeze(1).expand_as(m2_feature).bool()
            # eps = torch.zeros(1, dtype=m1_feature.dtype, device=m1_feature.device)
            # m1_feature = torch.where(m1_feature_masked , m1_feature, eps)
            # m2_feature = torch.where(m2_feature_masked , m2_feature, eps)
            if self.MaskType in ['AME', 'Shapley','AME_attention', 'Random']:
                self.m1_mask_num += (batch_size - m1_Mask.sum().item())
                self.m2_mask_num += (batch_size - m2_Mask.sum().item())
        
                m1_feature_masked = m1_Mask.unsqueeze(1).expand_as(m1_feature).bool()
                m2_feature_masked = m2_Mask.unsqueeze(1).expand_as(m2_feature).bool()
                eps = torch.zeros(1, dtype=m1_feature.dtype, device=m1_feature.device)
                m1_feature = torch.where(m1_feature_masked , m1_feature, eps)
                m2_feature = torch.where(m2_feature_masked , m2_feature, eps)
            elif self.MaskType == 'AME_attention':
                mask1_view = m1_Mask.view(-1, *([1]*(m1_feature.dim()-1)))
                mask2_view = m2_Mask.view(-1, *([1]*(m2_feature.dim()-1)))
                m1_feature = m1_feature * mask1_view
                m2_feature = m2_feature * mask2_view
            
    
            # 仅训练阶段会打印值
            if epoch_index == 0 and epoch != -1:
                print(f"{self.model_name[0]} mask_num is {self.m1_mask_num.item()}, {self.model_name[1]}_mask_num is {self.m2_mask_num.item()}")
                print(f"{self.model_name[0]} feature is {m1_feature}")
                print(f"{self.model_name[1]} feature is {m2_feature}")
        #     if self.MaskType == 'AME':
        #         kl_result,IT_gsd = self.Mask(
        #             m1_only_output,
        #             m2_only_output,
        #             labels=labels,
        #             epoch_index=epoch_index,
        #             epoch=epoch
        #         )
        #         # 计算掩码值
        #         m1_Mask = kl_result[:,0]
        #         m2_Mask = kl_result[:,1]
                
        #     elif self.MaskType == 'Shapley':
        #         m1_Mask, m2_Mask = self.Mask(
        #             m1_only_output,
        #             m2_only_output,
        #             fusion_logits=fusion_logits,
        #             labels=labels,
        #             epoch_index=epoch_index,
        #             epoch=epoch,
                    
        #         )
        #     if m1_Mask.dim() == 2:
        #         m1_Mask = m1_Mask.squeeze(1)
        #         m2_Mask = m2_Mask.squeeze(1)
        #     # 记录需要训练的强模态
        #     for i in range(batch_size):
        #         self.Mask_Dict_m1[sid[i]] = 0 if m1_Mask[i] else 1
        #         self.Mask_Dict_m2[sid[i]] = 0 if m2_Mask[i] else 1
        # # 训练强模态
        # elif epoch > self.args.warmup_epoch and self.Mask is not None and epoch%3==0:
        #     # 更新Mask的值 按照sid的计算
        #     for i in range(batch_size):
        #         m1_Mask[i] = self.Mask_Dict_m1[sid[i]]
        #         m2_Mask[i] = self.Mask_Dict_m2[sid[i]]
        
        # self.m1_mask_num += (batch_size - m1_Mask.sum().item())
        # self.m2_mask_num += (batch_size - m2_Mask.sum().item())

        # m1_feature_masked = m1_Mask.unsqueeze(1).expand_as(m1_feature).bool()
        # m2_feature_masked = m2_Mask.unsqueeze(1).expand_as(m2_feature).bool()
        # eps = torch.zeros(1, dtype=m1_feature.dtype, device=m1_feature.device)
        # m1_feature = torch.where(m1_feature_masked , m1_feature, eps)
        # m2_feature = torch.where(m2_feature_masked , m2_feature, eps)

        # # 仅训练阶段会打印值
        # if epoch_index == 0 and epoch != -1:
        #     print(f"{self.model_name[0]} mask_num is {self.m1_mask_num.item()}, {self.model_name[1]}_mask_num is {self.m2_mask_num.item()}")
        #     print(f"{self.model_name[0]} feature is {m1_feature}")
        #     print(f"{self.model_name[1]} feature is {m2_feature}")
            
        return m1_feature,m2_feature,m1_Mask,m2_Mask
    
        

# ==================基础融合方式====================
class ConcatFusion(base_model):
    def __init__(self, args):
        super(ConcatFusion, self).__init__(args)
        self.unified_dim = args.unified_dim
        self.args = args
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.unified_dim*2, 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )
        # self.fusion_classifier = nn.Sequential(
        #     nn.Linear(self.unified_dim * 2, self.outdim),
        #     nn.SiLU()
        #     # nn.SiLU(),
        #     # nn.Linear(64, self.outdim)
        # )

    def forward(self, 
                m1_feature, 
                m2_feature,
                labels=None,
                epoch=-1,
                epoch_index=-1,
                sid=None,
                ):
        if epoch_index == 0:
            self.m1_mask_num.zero_()
            self.m2_mask_num.zero_()
        # 直接拼接
        batch_size = m1_feature.shape[0]
        device = m1_feature.device
        IT_gsd = torch.full((batch_size, 2), 0.5, device=device) # 这个是贡献度的值，留作后续使用
        if self.model_name == ["Visual","Audio"] and m1_feature.dim() == 3:
            m1_feature = m1_feature.mean(dim=1)  # 全局平均池化
            m2_feature = m2_feature.mean(dim=1)  # 全局平均池化
        elif self.model_name == ["Image","Text"]:
            m1_feature = m1_feature[:,0,:]  # 取CLS token
            m2_feature = m2_feature[:,0,:]  # 取CLS token
        # if self.MaskType == 'AME':
        #     m1_feature,m2_feature,m1_Mask,m2_Mask = self.AME_MASK(epoch, batch_size, m1_feature, m2_feature, device, labels, epoch_index, sid=sid)
        # elif self.MaskType == 'Shapley':
        #     with torch.no_grad():
        #         fusion_feature = torch.cat((m1_feature, m2_feature), dim=-1)  # 在特征维度上拼接
        #         fusion_out = self.fusion_classifier(fusion_feature)
        #     m1_feature,m2_feature,m1_Mask,m2_Mask = self.AME_MASK(epoch, batch_size, m1_feature, m2_feature, device, labels, epoch_index, fusion_logits=fusion_out, sid=sid)
                # 缓存特征用于可视化
        # self.cached_m1_feature = m1_feature.detach().clone()
        # self.cached_m2_feature = m2_feature.detach().clone()
        # self.cached_fusion_feature = torch.cat((m1_feature, m2_feature), dim=-1).detach().clone()
        # print(f"cached_fusion_feature shape is {self.cached_fusion_feature.shape}")
        # with torch.no_grad():
        #     fusion_logits = self.fusion_classifier(self.cached_fusion_feature)
                # print(f"m2_feature is {m2_feature}")
        out_m1 = self.m1_classifier(m1_feature)
        out_m2 = self.m2_classifier(m2_feature)
        if self.use_OLA:
            with torch.no_grad():
                zero_feature = torch.zeros_like(m1_feature)
                fusion_m1_feature = torch.cat((m1_feature, zero_feature), dim=-1)
                fusion_m2_feature = torch.cat((zero_feature,m2_feature), dim=-1)
                m1_only_output = self.fusion_classifier(fusion_m1_feature)
                m2_only_output = self.fusion_classifier(fusion_m2_feature)
            m1_feature,m2_feature,m1_Mask,m2_Mask = self.AME_MASK(epoch, batch_size, m1_feature, m2_feature, device, labels, epoch_index, fusion_logits=None,sid=sid,m1_only_output=m1_only_output,m2_only_output=m2_only_output)
        else:
            m1_feature,m2_feature,m1_Mask,m2_Mask = self.AME_MASK(epoch, batch_size, m1_feature, m2_feature, device, labels, epoch_index, sid=sid)
        # m1_feature,m2_feature,m1_Mask,m2_Mask = self.AME_MASK(epoch, batch_size, m1_feature, m2_feature, device, labels, epoch_index, sid=sid)
        fusion_feature = torch.cat((m1_feature, m2_feature), dim=-1)  # 在特征维度上拼接
        fusion_out = self.fusion_classifier(fusion_feature)


        return fusion_out,out_m1,out_m2,m1_Mask,m2_Mask

class SumFusion(base_model):
    def __init__(self, args):
        super(SumFusion, self).__init__(args)
        self.args = args
        self.unified_dim = args.unified_dim
        
    def forward(self,  
                m1_feature, 
                m2_feature,
                labels=None,
                epoch=-1,
                epoch_index=-1,
                sid=None):
        if epoch_index == 0:
            self.m1_mask_num.zero_()
            self.m2_mask_num.zero_()
        # 直接拼接
        batch_size = m1_feature.shape[0]
        device = m1_feature.device
        IT_gsd = torch.full((batch_size, 2), 0.5, device=device) # 这个是贡献度的值，留作后续使用
        if self.model_name == ["Visual","Audio"] and m1_feature.dim() == 3:
            m1_feature = m1_feature.mean(dim=1)  # 全局平均池化
            m2_feature = m2_feature.mean(dim=1)  # 全局平均池化
        elif self.model_name == ["Image","Text"]:
            m1_feature = m1_feature[:,0,:]  # 取CLS token
            m2_feature = m2_feature[:,0,:]  # 取CLS token

        m1_feature,m2_feature,m1_Mask,m2_Mask = self.AME_MASK(epoch,batch_size,m1_feature,m2_feature,device,labels,epoch_index)
        out_m1 = self.m1_classifier(m1_feature)
        out_m2 = self.m2_classifier(m2_feature)
        fusion_out = out_m1 + out_m2
        # if self.use_OLA:
        #     # print(f"use ola")
        #     out_m1_new = self.m1_classifier(m1_feature)
        #     out_m2_new = self.m2_classifier(m2_feature)
        #     fusion_out = out_m1_new + out_m2_new
        # else:
        #     out_m1 = self.m1_classifier(m1_feature)
        #     out_m2 = self.m2_classifier(m2_feature)
        #     fusion_out = out_m1 + out_m2

        return fusion_out,out_m1,out_m2,m1_Mask,m2_Mask

class GatedFusion(base_model):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, args):
        super(GatedFusion, self).__init__(args)

        self.args = args
        self.unified_dim = args.unified_dim
        # self.fc_m1 = nn.Sequential(
        #     nn.Linear(self.unified_dim, 64),
        #     nn.SiLU(),
        #     nn.Linear(64, self.outdim)
        # )
        # self.fc_m2 = nn.Sequential(
        #     nn.Linear(self.unified_dim, 64),
        #     nn.SiLU(),
        #     nn.Linear(64, self.outdim)
        # )
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )
        self.fc_x = nn.Linear(self.unified_dim, self.unified_dim)
        self.fc_y = nn.Linear(self.unified_dim, self.unified_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, m1_feature,
            m2_feature,
            labels=None,
            epoch=-1,
            epoch_index=-1,
            sid=None):
        if epoch_index == 0:
            self.m1_mask_num.zero_()
            self.m2_mask_num.zero_()
        # 直接拼接
        batch_size = m1_feature.shape[0]
        device = m1_feature.device
        IT_gsd = torch.full((batch_size, 2), 0.5, device=device) # 这个是贡献度的值，留作后续使用
        if self.model_name == ["Visual","Audio"] and m1_feature.dim() == 3:
            m1_feature = m1_feature.mean(dim=1)  # 全局平均池化
            m2_feature = m2_feature.mean(dim=1)  # 全局平均池化
        elif self.model_name == ["Image","Text"]:
            m1_feature = m1_feature[:,0,:]  # 取CLS token
            m2_feature = m2_feature[:,0,:]  # 取CLS token
        m1_feature,m2_feature,m1_Mask,m2_Mask = self.AME_MASK(epoch,batch_size,m1_feature,m2_feature,device,labels,epoch_index)
        out_m1 = self.m1_classifier(m1_feature)
        out_m2 = self.m2_classifier(m2_feature)

        gate_x = self.sigmoid(self.fc_x(m1_feature))
        gate_y = self.sigmoid(self.fc_y(m2_feature))
        gate_sum = gate_x + gate_y + 1e-8
        gate_x = gate_x / gate_sum
        gate_y = gate_y / gate_sum
        fused_feature = gate_x * m1_feature + gate_y * m2_feature

        fusion_out = self.fusion_classifier(fused_feature)
        return fusion_out,out_m1,out_m2,m1_Mask,m2_Mask


class FiLM(base_model):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, args):
        super(FiLM, self).__init__(args)

        self.args = args
        self.fc = nn.Linear(self.unified_dim , 2 * self.unified_dim )
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.unified_dim , 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )
        # self.fc_m1 = nn.Sequential(
        #     nn.Linear(self.unified_dim, 64),
        #     nn.SiLU(),
        #     nn.Linear(64, self.outdim)
        # )
        # self.fc_m2 = nn.Sequential(
        #     nn.Linear(self.unified_dim, 64),
        #     nn.SiLU(),
        #     nn.Linear(64, self.outdim)
        # )
        self.fusion_classifier = nn.Linear(self.unified_dim , self.outdim)


    def forward(self,  
                m1_feature, 
                m2_feature,
                labels=None,
                epoch=-1,
                epoch_index=-1,
                sid=None):
        if epoch_index == 0:
           self.m1_mask_num.zero_()
           self.m2_mask_num.zero_()
        # 直接拼接
        batch_size = m1_feature.shape[0]
        device = m1_feature.device
        IT_gsd = torch.full((batch_size, 2), 0.5, device=device) # 这个是贡献度的值，留作后续使用
        if self.model_name == ["Visual","Audio"] and m1_feature.dim() == 3:
            m1_feature = m1_feature.mean(dim=1)  # 全局平均池化
            m2_feature = m2_feature.mean(dim=1)  # 全局平均池化
        elif self.model_name == ["Image","Text"]:
            m1_feature = m1_feature[:,0,:]  # 取CLS token
            m2_feature = m2_feature[:,0,:]  # 取CLS token
        m1_feature,m2_feature,m1_Mask,m2_Mask = self.AME_MASK(epoch,batch_size,m1_feature,m2_feature,device,labels,epoch_index)
        out_m1 = self.m1_classifier(m1_feature)
        out_m2 = self.m2_classifier(m2_feature)

        fused_feature = (m1_feature + m2_feature) / 2.0
        gamma,beta = torch.split(self.fc(fused_feature), self.unified_dim, 1)
        output = gamma * fused_feature + beta
        fusion_out = self.fusion_classifier(output)
        return fusion_out,out_m1,out_m2,m1_Mask,m2_Mask
        
# =============================三模态的使用===================================
class TVABaseModel(nn.Module):
    def __init__(self, args):
        super(TVABaseModel, self).__init__()
        self.batch_size = args.batch_size
        self.unified_dim = args.unified_dim
        self.outdim = args.num_classes
        self.args = args
        self.model_name = eval(args.model_name)

        # OPM 
        self.latest_drop_info = None
        self.opm_controller = None
        if getattr(args, 'use_opm', False) and len(self.model_name) >= 2:
            self.opm_controller = ModalityDropper(
                num_modalities=len(self.model_name),
                execute_prob=getattr(args, 'opm_execute_prob', 0.7),
                q_base=getattr(args, 'opm_q_base', 0.5),
                lam=getattr(args, 'opm_lambda', 0.5),
                min_keep=getattr(args, 'opm_min_keep', 1),
                seed=getattr(args, 'opm_seed', 2024),
            )
        # Random mask reproducibility: use an isolated CPU RNG stream.
        self._random_mask_gen = torch.Generator(device='cpu')
        seed = int(getattr(args, 'random_mask_seed', getattr(args, 'random_seed', 0)))
        self._random_mask_gen.manual_seed(seed)
        # 注册Mask Number 的buffer
        self.register_buffer('m1_mask_num', torch.tensor(0.0)) # 代表文本类
        self.register_buffer('m2_mask_num', torch.tensor(0.0)) # 代表视觉类
        self.register_buffer('m3_mask_num', torch.tensor(0.0)) # 代表音频类
        # 初始化AME模块
        self.MaskType = args.MaskType
        if self.MaskType == "None":
            self.Mask = None
        elif self.MaskType == "AME":
            self.Mask = AME(args=args)
        elif self.MaskType == "Shapley":
            self.Mask = Shapley(args=args)
        elif self.MaskType == "Random":
            self.Mask = None
        else:
            self.Mask == "None"
        self.ame_gap = args.ame_gap
        # 单模态分类器
        self.m1_classifier = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )
        self.m2_classifier = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )
        self.m3_classifier = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )

                # 存储Mask的状态
        self.Mask_Dict_m1 = defaultdict(lambda: torch.tensor(1, device='cuda:0'))
        self.Mask_Dict_m2 = defaultdict(lambda: torch.tensor(1, device='cuda:0'))
        self.Mask_Dict_m3 = defaultdict(lambda: torch.tensor(1, device='cuda:0'))
        

    def AME_MASK(self,epoch,batch_size,m1_feature,m2_feature,m3_feature,device,labels=None,epoch_index=-1, sid=None):
        if epoch_index == 0:
            self.m1_mask_num.zero_()
            self.m2_mask_num.zero_()
            self.m3_mask_num.zero_()
        drop_info = None
        if (
            self.opm_controller is not None
            and self.training
            and labels is not None
        ):
            raw_features = [m1_feature, m2_feature, m3_feature]
            with torch.no_grad():
                logits_for_drop = [
                    self.m1_classifier(raw_features[0].detach()),
                    self.m2_classifier(raw_features[1].detach()),
                    self.m3_classifier(raw_features[2].detach())
                ]
            dropped_features, drop_info = self.opm_controller(raw_features, logits_for_drop, labels)
            m1_feature, m2_feature, m3_feature = dropped_features
        self.latest_drop_info = drop_info
        m1_Mask =torch.ones(batch_size, device=device)
        m2_Mask =torch.ones(batch_size, device=device)
        m3_Mask =torch.ones(batch_size, device=device)
        if epoch > self.args.warmup_epoch and self.MaskType != "None" and epoch % self.ame_gap != 1:
            if self.MaskType == 'Random':
                drop_prob = float(getattr(self.args, 'random_mask_drop_prob', 0.0))
                masks = _sample_random_modality_masks(
                    batch_size=batch_size,
                    num_modalities=3,
                    drop_prob=drop_prob,
                    device=device,
                    dtype=m1_feature.dtype,
                    generator=self._random_mask_gen,
                )
                m1_Mask = masks[:, 0]
                m2_Mask = masks[:, 1]
                m3_Mask = masks[:, 2]
            else:
                with torch.no_grad():
                    m1_only_output = self.m1_classifier(m1_feature)
                    m2_only_output = self.m2_classifier(m2_feature)
                    m3_only_output = self.m3_classifier(m3_feature)
            if self.MaskType == 'AME':
                kl_result,IT_gsd = self.Mask(
                                        m1_only_output,
                                        m2_only_output,
                                        m3_only_output,
                                        labels=labels,
                                        epoch_index=epoch_index,
                                        epoch=epoch,)
                # 计算掩码值
                m1_Mask = kl_result[:,0]
                m2_Mask = kl_result[:,1]
                m3_Mask = kl_result[:,2]
                
            elif self.MaskType == 'Shapley':
                if self.args.fusion_method=='sum':
                    fusion_logits = m1_only_output + m2_only_output + m3_only_output
                m1_Mask, m2_Mask, m3_Mask = self.Mask(
                                        m1_only_output,
                                        m2_only_output,
                                        m3_only_output,
                                        fusion_logits=fusion_logits,
                                        labels=labels,
                                        epoch_index=epoch_index,
                                        epoch=epoch,)
            if m1_Mask.dim() == 2:
                m1_Mask = m1_Mask.squeeze(1)
                m2_Mask = m2_Mask.squeeze(1)
                m3_Mask = m3_Mask.squeeze(1)
            if self.MaskType in ['AME', 'Shapley', 'Random']:
                self.m1_mask_num += (batch_size - m1_Mask.sum().item())
                self.m2_mask_num += (batch_size - m2_Mask.sum().item())
                self.m3_mask_num += (batch_size - m3_Mask.sum().item())
        
                m1_feature_masked = m1_Mask.unsqueeze(1).expand_as(m1_feature).bool()
                m2_feature_masked = m2_Mask.unsqueeze(1).expand_as(m2_feature).bool()
                m3_feature_masked = m3_Mask.unsqueeze(1).expand_as(m3_feature).bool()
                eps = torch.zeros(1, dtype=m1_feature.dtype, device=m1_feature.device)
                m1_feature = torch.where(m1_feature_masked , m1_feature, eps)
                m2_feature = torch.where(m2_feature_masked , m2_feature, eps)
                m3_feature = torch.where(m3_feature_masked , m3_feature, eps)
    
            if epoch_index == 0:
                print(f"{self.model_name[0]} mask_num is {self.m1_mask_num.item()}, {self.model_name[1]}_mask_num is {self.m2_mask_num.item()}, {self.model_name[2]}_mask_num is {self.m3_mask_num.item()}")
                print(f"{self.model_name[0]} feature is {m1_feature}")
                print(f"{self.model_name[1]} feature is {m2_feature}")
                print(f"{self.model_name[2]} feature is {m3_feature}")
        # # 训练弱模态
        # if epoch > self.args.warmup_epoch and self.Mask is not None and epoch%3==2:
        # # if epoch > self.args.warmup_epoch and self.Mask is not None:
        #     with torch.no_grad():
        #         m1_only_output = self.m1_classifier(m1_feature)
        #         m2_only_output = self.m2_classifier(m2_feature)
        #         m3_only_output = self.m3_classifier(m3_feature)
        #     if self.MaskType == 'AME':
        #         kl_result,IT_gsd = self.Mask(
        #                                 m1_only_output,
        #                                 m2_only_output,
        #                                 m3_only_output,
        #                                 labels=labels,
        #                                 epoch_index=epoch_index,
        #                                 epoch=epoch,)
        #         # 计算掩码值
        #         m1_Mask = kl_result[:,0]
        #         m2_Mask = kl_result[:,1]
        #         m3_Mask = kl_result[:,2]
                
        #     elif self.MaskType == 'Shapley':
        #         if self.args.fusion_method=='sum':
        #             fusion_logits = m1_only_output + m2_only_output + m3_only_output
        #         m1_Mask, m2_Mask, m3_Mask = self.Mask(
        #                                 m1_only_output,
        #                                 m2_only_output,
        #                                 m3_only_output,
        #                                 fusion_logits=fusion_logits,
        #                                 labels=labels,
        #                                 epoch_index=epoch_index,
        #                                 epoch=epoch,)
        #     if m1_Mask.dim() == 2:
        #         m1_Mask = m1_Mask.squeeze(1)
        #         m2_Mask = m2_Mask.squeeze(1)
        #         m3_Mask = m3_Mask.squeeze(1)
        #     for i in range(batch_size):
        #         self.Mask_Dict_m1[sid[i]] = 0 if m1_Mask[i] else 1
        #         self.Mask_Dict_m2[sid[i]] = 0 if m2_Mask[i] else 1
        #         self.Mask_Dict_m3[sid[i]] = 0 if m3_Mask[i] else 1
        # # 训练强模态
        # elif epoch > self.args.warmup_epoch and self.Mask is not None and epoch%3==0:
        #     for i in range(batch_size):
        #         m1_Mask[i] = self.Mask_Dict_m1[sid[i]]
        #         m2_Mask[i] = self.Mask_Dict_m2[sid[i]]
        #         m3_Mask[i] = self.Mask_Dict_m3[sid[i]]
        # self.m1_mask_num += (batch_size - m1_Mask.sum().item())
        # self.m2_mask_num += (batch_size - m2_Mask.sum().item())
        # self.m3_mask_num += (batch_size - m3_Mask.sum().item())

        # m1_feature_masked = m1_Mask.unsqueeze(1).expand_as(m1_feature).bool()
        # m2_feature_masked = m2_Mask.unsqueeze(1).expand_as(m2_feature).bool()
        # m3_feature_masked = m3_Mask.unsqueeze(1).expand_as(m3_feature).bool()
        # eps = torch.zeros(1, dtype=m1_feature.dtype, device=m1_feature.device)
        # m1_feature = torch.where(m1_feature_masked , m1_feature, eps)
        # m2_feature = torch.where(m2_feature_masked , m2_feature, eps)
        # m3_feature = torch.where(m3_feature_masked , m3_feature, eps)

        # if epoch_index == 0:
        #     print(f"{self.model_name[0]} mask_num is {self.m1_mask_num.item()}, {self.model_name[1]}_mask_num is {self.m2_mask_num.item()}, {self.model_name[2]}_mask_num is {self.m3_mask_num.item()}")
        #     print(f"{self.model_name[0]} feature is {m1_feature}")
        #     print(f"{self.model_name[1]} feature is {m2_feature}")
        #     print(f"{self.model_name[2]} feature is {m3_feature}")
        return m1_feature,m2_feature,m3_feature,m1_Mask,m2_Mask,m3_Mask

class ConcatTVA(TVABaseModel):
    def __init__(self, args):  # 必须接收 args 参数
        super(ConcatTVA, self).__init__(args)

        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.unified_dim * 3, 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )
        
    def forward(self,
                m1_feature,
                m2_feature,
                m3_feature,
                labels=None,
                epoch=-1,
                epoch_index=-1,
                sid=None,
                ):

        batch_size = m1_feature.shape[0]
        device = m1_feature.device
        IT_gsd = torch.full((batch_size, 3), 0.5, device=device) # 这个是贡献度的值，留作后续使用
        # print("m1_feature shape :",m1_feature.shape)
        # print("m2_feature shape :",m2_feature.shape)
        # print("m3_feature shape :",m3_feature.shape)
        m1_feature = m1_feature.mean(dim=1)  # 全局平均池化
        m2_feature = m2_feature.mean(dim=1)  # 全局平均池化
        m3_feature = m3_feature.mean(dim=1)  # 全局平均池化
        # m1_feature = m1_feature[:,0,:] # 全局平均池化
        # m2_feature = m2_feature[:,0,:] # 全局平均池化
        # m3_feature = m3_feature[:,0,:] # 全局平均池化
        # if self.MaskType == 'AME':
        #     m1_feature,m2_feature,m3_feature,m1_Mask,m2_Mask,m3_Mask = self.AME_MASK(epoch,batch_size,m1_feature,m2_feature,m3_feature,device,labels,epoch_index,sid=sid)
        # elif self.MaskType == 'Shapley':
        #     with torch.no_grad():
        #         fusion_feature = torch.cat((m1_feature, m2_feature, m3_feature), dim=-1)  # 在特征维度上拼接
        #         fusion_out = self.fusion_classifier(fusion_feature)
        #     m1_feature,m2_feature,m3_feature,m1_Mask,m2_Mask,m3_Mask = self.AME_MASK(epoch,batch_size,m1_feature,m2_feature,m3_feature,device,labels,epoch_index, fusion_logits=fusion_out)
        m1_feature,m2_feature,m3_feature,m1_Mask,m2_Mask,m3_Mask = self.AME_MASK(epoch,batch_size,m1_feature,m2_feature,m3_feature,device,labels,epoch_index,sid=sid)
        fusion_feature = torch.cat((m1_feature, m2_feature, m3_feature), dim=-1)  # 在特征维度上拼接
        fusion_out = self.fusion_classifier(fusion_feature)
        out_m1 = self.m1_classifier(m1_feature)
        out_m2 = self.m2_classifier(m2_feature)
        out_m3 = self.m3_classifier(m3_feature)

        return fusion_out,out_m1,out_m2,out_m3,m1_Mask,m2_Mask,m3_Mask

class SumTVA(TVABaseModel):
    def __init__(self,args):
        super(SumTVA,self).__init__(args)
        self.args = args
    
    def forward(self,
                m1_feature,
                m2_feature,
                m3_feature,
                labels=None,
                epoch=-1,
                epoch_index=-1,
                sid=None,
                ):
        batch_size = m1_feature.shape[0]
        device = m1_feature.device
        IT_gsd = torch.full((batch_size, 3), 0.5, device=device) # 这个是贡献度的值，留作后续使用
        # print("m1_feature shape :",m1_feature.shape)
        # print("m2_feature shape :",m2_feature.shape)
        # print("m3_feature shape :",m3_feature.shape)
        # m1_feature = m1_feature.mean(dim=1)  # 全局平均池化
        # m2_feature = m2_feature.mean(dim=1)  # 全局平均池化
        # m3_feature = m3_feature.mean(dim=1)  # 全局平均池化
        m1_feature = m1_feature[:,0,:] # 全局平均池化
        m2_feature = m2_feature[:,0,:]  # 全局平均池化
        m3_feature = m3_feature[:,0,:]  # 全局平均池化
        m1_feature,m2_feature,m3_feature,m1_Mask,m2_Mask,m3_Mask = self.AME_MASK(epoch,batch_size,m1_feature,m2_feature,m3_feature,device,labels,epoch_index,sid=sid)
        out_m1 = self.m1_classifier(m1_feature)
        out_m2 = self.m2_classifier(m2_feature)
        out_m3 = self.m3_classifier(m3_feature)
        fusion_out = out_m1 + out_m2 + out_m3
        return fusion_out,out_m1,out_m2,out_m3,m1_Mask,m2_Mask,m3_Mask        


class GatedTVA(TVABaseModel):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """
    def __init__(self,args):
        super(GatedTVA,self).__init__(args)
        self.args = args
        self.fc_x = nn.Linear(self.unified_dim, self.unified_dim)
        self.fc_y = nn.Linear(self.unified_dim, self.unified_dim)
        self.fc_z = nn.Linear(self.unified_dim, self.unified_dim)

        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )

        self.sigmoid = nn.Sigmoid()
    def forward(self,
                m1_feature,
                m2_feature,
                m3_feature,
                labels=None,
                epoch=-1,
                epoch_index=-1,
                ):
        batch_size = m1_feature.shape[0]
        device = m1_feature.device
        IT_gsd = torch.full((batch_size, 3), 0.5, device=device) # 这个是贡献度的值，留作后续使用
        # print("m1_feature shape :",m1_feature.shape)
        # print("m2_feature shape :",m2_feature.shape)
        # print("m3_feature shape :",m3_feature.shape)
        m1_feature = m1_feature.mean(dim=1)  # 全局平均池化
        m2_feature = m2_feature.mean(dim=1)  # 全局平均池化
        m3_feature = m3_feature.mean(dim=1)  # 全局平均池化
        m1_feature,m2_feature,m3_feature,m1_Mask,m2_Mask,m3_Mask = self.AME_MASK(epoch,batch_size,m1_feature,m2_feature,m3_feature,device,labels,epoch_index)

        out_m1 = self.m1_classifier(m1_feature)
        out_m2 = self.m2_classifier(m2_feature)
        out_m3 = self.m3_classifier(m3_feature)

        gate_x = self.sigmoid(self.fc_x(m1_feature))
        gate_y = self.sigmoid(self.fc_y(m2_feature))
        gate_z = self.sigmoid(self.fc_z(m3_feature))
        gate_sum = gate_x + gate_y + gate_z + 1e-8  # 避免除零
        gate_x = gate_x / gate_sum
        gate_y = gate_y / gate_sum
        gate_z = gate_z / gate_sum
        fused_feature = gate_x * m1_feature + gate_y * m2_feature + gate_z * m3_feature

        fusion_out = self.fusion_classifier(fused_feature)

        return fusion_out, out_m1, out_m2, out_m3, m1_Mask, m2_Mask, m3_Mask

class FiLMTVA(TVABaseModel):
    def __init__(self,args):
        super(FiLMTVA,self).__init__(args)
        self.args = args
        self.fc = nn.Linear(self.unified_dim , 2 * self.unified_dim )

        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.unified_dim , 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )
    def forward(self,
                m1_feature,
                m2_feature,
                m3_feature,
                labels=None,
                epoch=-1,
                epoch_index=-1,
                ):
        batch_size = m1_feature.shape[0]
        device = m1_feature.device
        IT_gsd = torch.full((batch_size, 3), 0.5, device=device) # 这个是贡献度的值，留作后续使用
        # print("m1_feature shape :",m1_feature.shape)
        # print("m2_feature shape :",m2_feature.shape)
        # print("m3_feature shape :",m3_feature.shape)
        m1_feature = m1_feature.mean(dim=1)  # 全局平均池化
        m2_feature = m2_feature.mean(dim=1)  # 全局平均池化
        m3_feature = m3_feature.mean(dim=1)  # 全局平均池化
        m1_feature,m2_feature,m3_feature,m1_Mask,m2_Mask,m3_Mask = self.AME_MASK(epoch,batch_size,m1_feature,m2_feature,m3_feature,device,labels,epoch_index)

        out_m1 = self.m1_classifier(m1_feature)
        out_m2 = self.m2_classifier(m2_feature)
        out_m3 = self.m3_classifier(m3_feature)

        fused_feature = (m1_feature + m2_feature + m3_feature) / 3.0

        gamma, beta = torch.split(self.fc(fused_feature), self.unified_dim, 1)

        output = gamma * fused_feature + beta

        fusion_out = self.fusion_classifier(output)

        return fusion_out,out_m1,out_m2,out_m3,m1_Mask,m2_Mask,m3_Mask

# =============================MLA组件================================
class ConcatFusion_MLA(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(ConcatFusion_MLA, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output

class ConcatFusion3_MLA(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(ConcatFusion3_MLA, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y, z):
        output = torch.cat((x, y, z), dim=1)
        
        output = self.fc_out(output)
        return x, y, z, output


# =============================注意力机制融合================================
class CAfusion(base_model):
    def __init__(self, args):
        super(CAfusion, self).__init__(args)
        # token 长度（命名为 visual/audio 以便 forward 中使用）
        self.visual_token_len = args.m1_token_len
        self.audio_token_len = args.m2_token_len
        # 位置编码器：延后在 forward 中用统一的生成器创建（避免构造签名差异）

        self.co_attention = CoAttentionExpert(dim=self.unified_dim, num_heads=8)
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.unified_dim , self.unified_dim),
            nn.SiLU(),
            nn.Linear(self.unified_dim, self.outdim)
        )
        self.m1_classifier = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )
        self.m2_classifier = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )
        self.final_fusing_experts = Block(dim=self.unified_dim,num_heads=8)

    def forward(self, 
                m1_feature, 
                m2_feature,
                labels=None,
                epoch=-1,
                epoch_index=-1,
                sid=None
                ):
        if epoch_index == 0:
            self.m1_mask_num.zero_()
            self.m2_mask_num.zero_()
        device = m1_feature.device
        batch_size = m1_feature.shape[0]
        # 生成位置编码的简单方式：PositionalEncoding1D(d_model) 接受 (B, L, C) 输入并返回同形状张量
        pe_gen = PositionalEncoding1D(self.unified_dim).to(device)
        pos_visual = pe_gen(torch.zeros(batch_size, self.visual_token_len, self.unified_dim, device=device))
        pos_audio  = pe_gen(torch.zeros(batch_size, self.audio_token_len,  self.unified_dim, device=device))
        # 我们将四个特征看作序列长度为 4 的输入
        pos_fusion = pe_gen(torch.zeros(batch_size, 4, self.unified_dim, device=device))

        IT_gsd = torch.full((batch_size, 2), 0.5, device=device) # 这个是贡献度的值，留作后续使用
        if self.model_name == ["Visual","Audio"] and m1_feature.dim() == 3:
            m1_feature = m1_feature.mean(dim=1)  # 全局平均池化
            m2_feature = m2_feature.mean(dim=1)  # 全局平均池化
        elif self.model_name == ["Image","Text"]:
            m1_feature = m1_feature[:,0,:]  # 取CLS token
            m2_feature = m2_feature[:,0,:]  # 取CLS token
        # m1_Mask =torch.ones(batch_size, device=device)
        # m2_Mask =torch.ones(batch_size, device=device)
        # # ================AME掩码================
        # if epoch > self.args.warmup_epoch and self.Mask is not None and epoch%2==0:
        #     with torch.no_grad():
        #         # shared_visual_feature_lite = self.m1_trim(m1_feature)
        #         # shared_audio_feature_lite = self.m2_trim(m2_feature)
        #         # visual_only_output = self.m1_alone_classifier(shared_visual_feature_lite)
        #         # audio_only_output = self.m2_alone_classifier(shared_audio_feature_lite)
        #         visual_only_output = self.m1_classifier(m1_feature)
        #         audio_only_output = self.m2_classifier(m2_feature)
        #     kl_result,IT_gsd = self.Mask(
        #         p=visual_only_output,
        #         q=audio_only_output,
        #         labels=labels,
        #         epoch_index=epoch_index,
        #         epoch=epoch
        #     )
        #     # 计算掩码值（kl_result 期望形状为 (B,2)）
        #     m1_Mask = kl_result[:,0]
        #     m2_Mask = kl_result[:,1]
        #     # 若当前为 (B,)，将其扩展为 (B,1) 以便后续 expand_as 使用
        #     if m1_Mask.dim() == 1:
        #         m1_Mask = m1_Mask.unsqueeze(1)
        #         m2_Mask = m2_Mask.unsqueeze(1)
        #     # 以 buffer 形式记录被掩掉的样本数（使用 in-place 操作以保持 buffer）
        #     self.m1_mask_num.zero_() if epoch_index == 0 else None
        #     self.m2_mask_num.zero_() if epoch_index == 0 else None
        #     self.m1_mask_num += (batch_size - m1_Mask.sum().item())
        #     self.m2_mask_num += (batch_size - m2_Mask.sum().item())

        #     m1_feature_masked = m1_Mask.unsqueeze(1).expand_as(m1_feature) 
        #     m2_feature_masked = m2_Mask.unsqueeze(1).expand_as(m2_feature)
        #     eps = torch.zeros(1, dtype=m1_feature.dtype, device=m1_feature.device)
        #     m1_feature = torch.where(m1_feature_masked , m1_feature, eps)
        #     m2_feature = torch.where(m2_feature_masked , m2_feature, eps)

        #     if epoch_index == 0:
        #         print(f"{self.model_name[0]} mask_num is {self.m1_mask_num.item()}, {self.model_name[1]}_mask_num is {self.m2_mask_num.item()}")
        #         print(f"{self.model_name[0]} feature is {m1_feature}")
        #         print(f"{self.model_name[1]} feature is {m2_feature}")
        m1_feature,m2_feature,m1_Mask,m2_Mask = self.AME_MASK(epoch, batch_size, m1_feature, m2_feature, device, labels, epoch_index, sid=sid)
        
        # 有参数融合
        shared_image_mm_features, shared_text_mm_features = self.co_attention(
            m1_feature.unsqueeze(1) + pos_visual,
            m2_feature.unsqueeze(1) + pos_audio
        )
        shared_visual_mm_feature = shared_image_mm_features.mean(dim=1)
        shared_audio_mm_feature = shared_text_mm_features.mean(dim=1)

        # 将四个向量视为长度为 4 的序列（每项维度为 unified_dim），方便用 Transformer Block 进行交互
        # 形状：(B, 4, unified_dim)
        fusion_sequence = torch.stack((
            m1_feature,
            shared_visual_mm_feature,
            shared_audio_mm_feature,
            m2_feature
        ), dim=1)

        # 加上位置编码并通过最终的 Transformer Block（timml Block 接受 (B, N, C)）
        fusion_after = self.final_fusing_experts(fusion_sequence + pos_fusion)

        # 池化为单个向量用于分类
        final_fusion_feature = fusion_after.mean(dim=1)  # (B, unified_dim)
        # final_fusion_feature = fusion_after[:,0]  # (B, unified_dim)
        fusion_out = self.fusion_classifier(final_fusion_feature)
        out_m1 = self.m1_classifier(m1_feature)
        out_m2 = self.m2_classifier(m2_feature)

        # 返回格式与 ConcatFusion 保持一致，以便上层调用方解包为 (fusion_out, out1, out2, m1_mask, m2_mask)
        return fusion_out, out_m1, out_m2, m1_Mask, m2_Mask





# =================MMTM (Multi-Modal Transfer Module)========================
class MMTMModule(nn.Module):
    """
    MMTM: Multi-Modal Transfer Module
    参考论文: Learning Relationships for Multi-View 3D Object Recognition (ICCV 2019)
    通过挤压-激励机制实现跨模态特征调制
    """
    def __init__(self, dim_visual, dim_audio, ratio=4):
        super(MMTMModule, self).__init__()
        dim = dim_visual + dim_audio
        dim_out = int(2 * dim / ratio)
        
        # 挤压层：将两个模态的信息融合
        self.fc_squeeze = nn.Linear(dim, dim_out)
        
        # 激励层：生成每个模态的调制权重
        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_audio = nn.Linear(dim_out, dim_audio)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, visual, audio):
        """
        Args:
            visual: 视觉特征 (B, C_v) 或 (B, C_v, H, W)
            audio: 音频特征 (B, C_a) 或 (B, C_a, H, W)
        Returns:
            调制后的视觉和音频特征
        """
        # 如果是多维特征，先进行全局平均池化得到通道级统计信息
        squeeze_array = []
        for tensor in [visual, audio]:
            if len(tensor.shape) > 2:
                tview = tensor.view(tensor.shape[:2] + (-1,))
                squeeze_array.append(torch.mean(tview, dim=-1))
            else:
                squeeze_array.append(tensor)
        
        # 拼接两个模态的特征
        squeeze = torch.cat(squeeze_array, 1)
        
        # 通过挤压层学习跨模态关系
        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)
        
        # 为每个模态生成调制权重
        vis_out = self.fc_visual(excitation)
        aud_out = self.fc_audio(excitation)
        
        vis_out = self.sigmoid(vis_out)
        aud_out = self.sigmoid(aud_out)
        
        # 根据特征维度扩展权重
        if len(visual.shape) > 2:
            dim_diff = len(visual.shape) - len(vis_out.shape)
            vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)
        
        if len(audio.shape) > 2:
            dim_diff = len(audio.shape) - len(aud_out.shape)
            aud_out = aud_out.view(aud_out.shape + (1,) * dim_diff)
        
        # 应用调制权重
        return visual * vis_out, audio * aud_out


class MMTMFusion(base_model):
    """
    基于MMTM的多模态融合模型
    在多个特征层级上使用MMTM模块进行跨模态调制
    """
    def __init__(self, args):
        super(MMTMFusion, self).__init__(args)
        self.args = args
        self.unified_dim = args.unified_dim
        
        # 多层MMTM模块，在不同层级进行特征交互
        self.mmtm1 = MMTMModule(
            dim_visual=self.unified_dim,
            dim_audio=self.unified_dim,
            ratio=4
        )
        self.mmtm2 = MMTMModule(
            dim_visual=self.unified_dim,
            dim_audio=self.unified_dim,
            ratio=4
        )
        self.mmtm3 = MMTMModule(
            dim_visual=self.unified_dim,
            dim_audio=self.unified_dim,
            ratio=4
        )
        
        # 特征变换层
        self.transform1 = nn.Sequential(
            nn.Linear(self.unified_dim, self.unified_dim),
            nn.LayerNorm(self.unified_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.transform2 = nn.Sequential(
            nn.Linear(self.unified_dim, self.unified_dim),
            nn.LayerNorm(self.unified_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 融合分类器
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.unified_dim * 2, 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )

    def forward(self, 
                m1_feature, 
                m2_feature,
                labels=None,
                epoch=-1,
                epoch_index=-1,
                sid=None):
        if epoch_index == 0:
            self.m1_mask_num.zero_()
            self.m2_mask_num.zero_()
        
        batch_size = m1_feature.shape[0]
        device = m1_feature.device
        
        # 处理特征格式
        if self.model_name == ["Visual", "Audio"] and m1_feature.dim() == 3:
            m1_feature = m1_feature.mean(dim=1)  # 全局平均池化
            m2_feature = m2_feature.mean(dim=1)
        elif self.model_name == ["Image", "Text"]:
            m1_feature = m1_feature[:, 0, :]  # 取CLS token
            m2_feature = m2_feature[:, 0, :]
        
        # 保存原始特征用于后续处理
        m1_orig = m1_feature.clone()
        m2_orig = m2_feature.clone()
        
        # 第一层MMTM：初始跨模态调制
        m1_mod1, m2_mod1 = self.mmtm1(m1_feature, m2_feature)
        
        # 特征变换并添加残差连接
        m1_trans1 = self.transform1(m1_mod1) + m1_orig
        m2_trans1 = self.transform1(m2_mod1) + m2_orig
        
        # 第二层MMTM：深度跨模态交互
        m1_mod2, m2_mod2 = self.mmtm2(m1_trans1, m2_trans1)
        
        # 特征变换并添加残差连接
        m1_trans2 = self.transform2(m1_mod2) + m1_trans1
        m2_trans2 = self.transform2(m2_mod2) + m2_trans1
        
        # 第三层MMTM：最终特征对齐
        m1_final, m2_final = self.mmtm3(m1_trans2, m2_trans2)
        
        # 应用AME Mask（在最终特征上）
        m1_final, m2_final, m1_Mask, m2_Mask = self.AME_MASK(
            epoch, batch_size, m1_final, m2_final, device, labels, epoch_index, sid=sid
        )
        
        # 融合两个模态
        fusion_feature = torch.cat([m1_final, m2_final], dim=-1)
        fusion_out = self.fusion_classifier(fusion_feature)
        
        # 单模态分类输出
        out_m1 = self.m1_classifier(m1_final)
        out_m2 = self.m2_classifier(m2_final)
        
        return fusion_out, out_m1, out_m2, m1_Mask, m2_Mask


class MMTMFusionSimple(base_model):
    """
    简化版MMTM融合模型
    只使用单层MMTM进行跨模态调制
    """
    def __init__(self, args):
        super(MMTMFusionSimple, self).__init__(args)
        self.args = args
        self.unified_dim = args.unified_dim
        
        # 单个MMTM模块
        self.mmtm = MMTMModule(
            dim_visual=self.unified_dim,
            dim_audio=self.unified_dim,
            ratio=4
        )
        
        # 融合分类器
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.unified_dim * 2, 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )

    def forward(self, 
                m1_feature, 
                m2_feature,
                labels=None,
                epoch=-1,
                epoch_index=-1,
                sid=None):
        if epoch_index == 0:
            self.m1_mask_num.zero_()
            self.m2_mask_num.zero_()
        
        batch_size = m1_feature.shape[0]
        device = m1_feature.device
        
        # 处理特征格式
        if self.model_name == ["Visual", "Audio"] and m1_feature.dim() == 3:
            m1_feature = m1_feature.mean(dim=1)
            m2_feature = m2_feature.mean(dim=1)
        elif self.model_name == ["Image", "Text"]:
            m1_feature = m1_feature[:, 0, :]
            m2_feature = m2_feature[:, 0, :]
        
        # 应用MMTM进行跨模态调制
        m1_modulated, m2_modulated = self.mmtm(m1_feature, m2_feature)
        
        # 应用AME Mask
        m1_modulated, m2_modulated, m1_Mask, m2_Mask = self.AME_MASK(
            epoch, batch_size, m1_modulated, m2_modulated, device, labels, epoch_index, sid=sid
        )
        
        # 融合并分类
        fusion_feature = torch.cat([m1_modulated, m2_modulated], dim=-1)
        fusion_out = self.fusion_classifier(fusion_feature)
        
        # 单模态分类输出
        out_m1 = self.m1_classifier(m1_modulated)
        out_m2 = self.m2_classifier(m2_modulated)
        
        return fusion_out, out_m1, out_m2, m1_Mask, m2_Mask


# =================CentralNet========================
class FusionBlock(nn.Module):
    """
    CentralNet的融合块，用于逐层融合两个模态的特征
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dropout_prob=0.1,
        fusion_method="add",
    ):
        super(FusionBlock, self).__init__()
        
        if fusion_method == "add":
            self.fusion_method = lambda x1, x2: torch.add(x1, x2)
        elif fusion_method == "mul":
            self.fusion_method = lambda x1, x2: torch.mul(x1, x2)
        elif fusion_method == "concat":
            self.fusion_method = lambda x1, x2: torch.cat((x1, x2), dim=-1)
            in_channels = in_channels * 2  # concat会使通道数翻倍
        else:
            self.fusion_method = lambda x1, x2: torch.add(x1, x2)
        
        self.fusion_fc = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x1, x2, x_central=None):
        """
        Args:
            x1: 模态1的特征 (B, L1, C)
            x2: 模态2的特征 (B, L2, C)
            x_central: 上一层的中心融合特征 (B, L, C)
        """
        # 如果序列长度不同，需要对齐
        if x1.shape[1] != x2.shape[1]:
            # 使用平均池化对齐到较短的长度
            target_len = min(x1.shape[1], x2.shape[1])
            if x1.shape[1] > target_len:
                x1 = F.adaptive_avg_pool1d(x1.transpose(1, 2), target_len).transpose(1, 2)
            if x2.shape[1] > target_len:
                x2 = F.adaptive_avg_pool1d(x2.transpose(1, 2), target_len).transpose(1, 2)
        
        # 融合两个模态
        if x_central is None:
            x = self.fusion_method(x1, x2)
        else:
            # 对齐中心特征的长度
            if x_central.shape[1] != x1.shape[1]:
                x_central = F.adaptive_avg_pool1d(
                    x_central.transpose(1, 2), x1.shape[1]
                ).transpose(1, 2)
            x = x_central + self.fusion_method(x1, x2)
        
        x = self.fusion_fc(x)
        x = self.relu(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class CentralNetFusion(base_model):
    """
    基于CentralNet的融合模型，通过多层级的FusionBlock逐步融合两个模态
    参考论文: CentralNet - https://arxiv.org/abs/xxxx
    """
    def __init__(self, args):
        super(CentralNetFusion, self).__init__(args)
        self.args = args
        self.unified_dim = args.unified_dim
        
        # 定义多层FusionBlock，逐步融合特征
        self.fusion_layer1 = FusionBlock(
            in_channels=self.unified_dim,
            out_channels=self.unified_dim,
            stride=1,
            dropout_prob=0.1,
            fusion_method="add"
        )
        self.fusion_layer2 = FusionBlock(
            in_channels=self.unified_dim,
            out_channels=self.unified_dim,
            stride=1,
            dropout_prob=0.1,
            fusion_method="add"
        )
        self.fusion_layer3 = FusionBlock(
            in_channels=self.unified_dim,
            out_channels=self.unified_dim,
            stride=1,
            dropout_prob=0.1,
            fusion_method="add"
        )
        
        # 使用Transformer Block进一步处理融合特征
        self.fusion_transformer = Block(dim=self.unified_dim, num_heads=8)
        
        # 最终分类器
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.unified_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.outdim)
        )

    def forward(self, 
                m1_feature, 
                m2_feature,
                labels=None,
                epoch=-1,
                epoch_index=-1,
                sid=None):
        if epoch_index == 0:
            self.m1_mask_num.zero_()
            self.m2_mask_num.zero_()
        
        batch_size = m1_feature.shape[0]
        device = m1_feature.device
        
        # 处理不同模态的特征格式
        if self.model_name == ["Visual", "Audio"]:
            # 保持序列格式用于逐层融合
            if m1_feature.dim() == 2:
                m1_feature = m1_feature.unsqueeze(1)
            if m2_feature.dim() == 2:
                m2_feature = m2_feature.unsqueeze(1)
            
            # 为单模态分类器提取全局特征
            m1_global = m1_feature.mean(dim=1)
            m2_global = m2_feature.mean(dim=1)
            
        elif self.model_name == ["Image", "Text"]:
            # 保持序列格式
            if m1_feature.dim() == 2:
                m1_feature = m1_feature.unsqueeze(1)
            if m2_feature.dim() == 2:
                m2_feature = m2_feature.unsqueeze(1)
            
            # 提取CLS token用于单模态分类
            m1_global = m1_feature[:, 0, :]
            m2_global = m2_feature[:, 0, :]
        else:
            # 默认处理
            if m1_feature.dim() == 2:
                m1_feature = m1_feature.unsqueeze(1)
            if m2_feature.dim() == 2:
                m2_feature = m2_feature.unsqueeze(1)
            m1_global = m1_feature.mean(dim=1)
            m2_global = m2_feature.mean(dim=1)
        
        # 应用AME Mask
        m1_global, m2_global, m1_Mask, m2_Mask = self.AME_MASK(
            epoch, batch_size, m1_global, m2_global, device, labels, epoch_index, sid=sid
        )
        
        # 将mask应用到序列特征上
        if m1_Mask is not None:
            m1_mask_expanded = m1_Mask.unsqueeze(1).unsqueeze(2).expand_as(m1_feature)
            m2_mask_expanded = m2_Mask.unsqueeze(1).unsqueeze(2).expand_as(m2_feature)
            m1_feature = m1_feature * m1_mask_expanded
            m2_feature = m2_feature * m2_mask_expanded
        
        # 逐层融合特征
        central = self.fusion_layer1(m1_feature, m2_feature, x_central=None)
        central = self.fusion_layer2(m1_feature, m2_feature, x_central=central)
        central = self.fusion_layer3(m1_feature, m2_feature, x_central=central)
        
        # 使用Transformer进一步处理融合特征
        central = self.fusion_transformer(central)
        
        # 全局池化得到最终的融合特征
        fusion_feature = central.mean(dim=1)
        
        # 分类输出
        fusion_out = self.fusion_classifier(fusion_feature)
        out_m1 = self.m1_classifier(m1_global)
        out_m2 = self.m2_classifier(m2_global)
        
        return fusion_out, out_m1, out_m2, m1_Mask, m2_Mask





