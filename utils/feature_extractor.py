"""
特征提取工具
用于在训练过程中保存模态特征，以便后续可视化
"""

import torch
import numpy as np
import os
from typing import Dict, List, Tuple
import pickle


class FeatureCollector:
    """特征收集器，用于在训练过程中收集和保存特征"""
    
    def __init__(self, save_dir: str, modality_names: List[str]):
        """
        Args:
            save_dir: 特征保存目录
            modality_names: 模态名称列表，如 ["Visual", "Audio"] 或 ["Image", "Text"]
        """
        self.save_dir = save_dir
        self.modality_names = modality_names
        self.features_cache = {}
        
        os.makedirs(save_dir, exist_ok=True)
        print(f"FeatureCollector initialized. Save dir: {save_dir}")
        print(f"Modalities: {modality_names}")
    
    def collect_features(self, 
                        epoch: int, 
                        fusion_features: torch.Tensor,
                        modality_features: List[torch.Tensor],
                        labels: torch.Tensor):
        """
        收集当前batch的特征
        
        Args:
            epoch: 当前epoch
            fusion_features: 融合特征 (B, D)
            modality_features: 各模态特征列表 [(B, D), (B, D), ...]
            labels: 标签 (B,)
        """
        if epoch not in self.features_cache:
            self.features_cache[epoch] = {
                'fusion': [],
                'labels': []
            }
            for name in self.modality_names:
                self.features_cache[epoch][name] = []
        
        # 保存特征到CPU并转为numpy
        self.features_cache[epoch]['fusion'].append(fusion_features.detach().cpu().numpy())
        self.features_cache[epoch]['labels'].append(labels.detach().cpu().numpy())
        
        for name, feat in zip(self.modality_names, modality_features):
            self.features_cache[epoch][name].append(feat.detach().cpu().numpy())
    
    def save_epoch_features(self, epoch: int):
        """
        保存某个epoch的所有特征到文件
        
        Args:
            epoch: 要保存的epoch
        """
        if epoch not in self.features_cache:
            print(f"Warning: No features collected for epoch {epoch}")
            return
        
        # 合并所有batch的特征
        epoch_data = {}
        epoch_data['fusion'] = np.concatenate(self.features_cache[epoch]['fusion'], axis=0)
        epoch_data['labels'] = np.concatenate(self.features_cache[epoch]['labels'], axis=0)
        
        for name in self.modality_names:
            epoch_data[name] = np.concatenate(self.features_cache[epoch][name], axis=0)
        
        # 保存到文件
        save_path = os.path.join(self.save_dir, f'features_epoch_{epoch}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(epoch_data, f)
        
        print(f"Saved features for epoch {epoch} to {save_path}")
        print(f"  Fusion features shape: {epoch_data['fusion'].shape}")
        print(f"  Labels shape: {epoch_data['labels'].shape}")
        for name in self.modality_names:
            print(f"  {name} features shape: {epoch_data[name].shape}")
        
        # 清理cache以节省内存
        del self.features_cache[epoch]
    
    def load_epoch_features(self, epoch: int) -> Dict:
        """
        加载某个epoch的特征
        
        Args:
            epoch: 要加载的epoch
            
        Returns:
            包含特征和标签的字典
        """
        load_path = os.path.join(self.save_dir, f'features_epoch_{epoch}.pkl')
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Features file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        return data


def extract_features_from_model(model, dataloader, device, modality_names: List[str], 
                               max_samples: int = 5000) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    从训练好的模型中提取特征（用于事后分析）
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        modality_names: 模态名称列表
        max_samples: 最多提取的样本数
        
    Returns:
        features_dict: 包含各模态特征的字典
        labels: 标签数组
    """
    model.eval()
    
    features_dict = {
        'fusion': [],
        **{name: [] for name in modality_names}
    }
    labels_list = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= max_samples:
                break
            
            # 根据模态组合解析数据
            if modality_names == ["Visual", "Audio"]:
                spec, image, label = batch[0], batch[1], batch[2]
                spec, image, label = spec.to(device), image.to(device), label.to(device)
                data_packet = (image.float(), spec.unsqueeze(1).float())
            elif modality_names == ["Image", "Text"]:
                token, padding_mask, image, label = batch[0], batch[1], batch[2], batch[3]
                token, padding_mask = token.to(device), padding_mask.to(device)
                image, label = image.to(device), label.to(device)
                data_packet = (token, padding_mask, image)
            elif modality_names == ["Text", "Visual", "Audio"]:
                token, padding_mask, image, spec, label = batch[0], batch[1], batch[2], batch[3], batch[4]
                token, padding_mask = token.to(device), padding_mask.to(device)
                image, spec, label = image.to(device), spec.to(device), label.to(device)
                data_packet = (token, padding_mask, image.float(), spec.unsqueeze(1).float())
            else:
                raise ValueError(f"Unsupported modality combination: {modality_names}")
            
            # 前向传播获取特征
            outputs = model(data_packet)
            
            # 提取各层特征
            # outputs通常是 (fusion_logits, m1_logits, m2_logits, [m3_logits], ...)
            fusion_logits = outputs[0]
            modality_logits = outputs[1:len(modality_names)+1]
            
            # 保存特征
            features_dict['fusion'].append(fusion_logits.cpu().numpy())
            labels_list.append(label.cpu().numpy())
            
            for name, logits in zip(modality_names, modality_logits):
                features_dict[name].append(logits.cpu().numpy())
            
            sample_count += label.size(0)
    
    # 合并所有batch
    for key in features_dict:
        features_dict[key] = np.concatenate(features_dict[key], axis=0)[:max_samples]
    labels = np.concatenate(labels_list, axis=0)[:max_samples]
    
    return features_dict, labels
