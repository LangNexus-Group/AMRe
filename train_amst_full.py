"""
AMST (Asynchronous Multi-modal Skip Training) Implementation
Based on train_all.py framework
"""
import os
import csv
import ast
import time
import torch
import random
import datetime
import argparse

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.dataloader import AV_CD_Dataset
from model.basic_model import VA_Classifier, TVA_Classifier
from dataset.Mydataset import TVADataset, M3AEDataset
from dataset.Mydataset import CramedDataset, AVEDataset, KSDataset

from utils.metrics import calculate_metrics
from utils.utils import setup_seed, weight_init, print_model_params


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str, help='KineticSound, CREMAD, AVE, MVSA, Food101, IEMOCAP3')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--train', action='store_true', help='turn on train mode')
    parser.add_argument('--use_video_frames', default=3, type=int, help='use how many frames for train')
    
    # 学习率定义
    parser.add_argument('--learning_rate_visual', default=1e-3, type=float, help='Learning rate for visual encoder')
    parser.add_argument('--learning_rate_audio', default=1e-3, type=float, help='Learning rate for audio encoder')
    parser.add_argument('--learning_rate_text', default=1e-5, type=float, help='Learning rate for text encoder')
    parser.add_argument('--learning_rate_image', default=1e-3, type=float, help='Learning rate for image encoder')
    parser.add_argument('--learning_rate_fusion', default=1e-3, type=float, help='Learning rate for fusion model')
    parser.add_argument('--optimizer', default='Adamw', type=str, help='Optimizer to use (SGD, Adam, Adamw)')
    parser.add_argument('--lr_decay_step', default=30, type=int)
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    
    # 基础设置
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU ids')
    parser.add_argument('--current_epoch', type=int, default=1, help="Start train epoch number")
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    
    # 模型设置
    parser.add_argument('--fusion_method', default='concat', type=str,
                       choices=['sum', 'concat', 'Gate', 'Film', 'share', 'CAfusion'],
                       help='Fusion method (will use both sum and concat for AMST)')
    parser.add_argument('--model_name', default='["Visual","Audio"]', type=str,
                       choices=['["Visual","Audio"]', '["Image","Text"]', '["Text","Visual","Audio"]'])
    parser.add_argument('--modality', default='full', type=str,
                       choices=['full', 'audio', 'visual'],
                       help='modality to use')
    parser.add_argument('--unified_dim', default=512, type=int, help='Unified feature dimension after encoders')
    parser.add_argument('--m1_token_len', default=1, type=int, help='Modality 1 token length')
    parser.add_argument('--m2_token_len', default=1, type=int, help='Modality 2 token length')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes')
    
    # AMST特定参数
    parser.add_argument('--use_amst', default=True, type=bool, help='Whether to use AMST training')
    parser.add_argument('--skip_factor_audio', default=1, type=int, help='Skip factor for audio (1=no skip)')
    parser.add_argument('--skip_factor_visual', default=1, type=int, help='Skip factor for visual (1=no skip)')
    parser.add_argument('--skip_factor_text', default=1, type=int, help='Skip factor for text (1=no skip)')
    parser.add_argument('--use_helper', default=False, type=bool, help='Whether to use helper classifier')
    parser.add_argument('--helper_weight', default=0.5, type=float, help='Weight for helper loss')
    parser.add_argument('--alpha', default=1.0, type=float, help='Weight for single modality losses')
    
    # AME相关参数（保持兼容）
    parser.add_argument('--MaskType', default='None', type=str, choices=['None', 'AME'], help='Type of masking strategy')
    parser.add_argument('--ame_gama', default=0.1, type=float, help='Gamma parameter for AME module')
    parser.add_argument('--ame_beta', default=0.7, type=float, help='Beta parameter for AME module')
    parser.add_argument('--ame_gap', default=2, type=int, help='restoration gap')
    parser.add_argument('--ame_gap_start', default=1, type=int, help='restoration gap start')
    parser.add_argument('--ame_temperature', default=0.2, type=float, help='Temperature parameter for AME module')
    parser.add_argument('--warmup_epoch', default=0, type=int, help='Number of warmup epochs before applying AME')
    parser.add_argument('--Use_MACE', default=True, type=bool, help="是否使用MACE loss")
    
    # tensorboard相关参数
    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')
    
    # 保存模型的相关参数
    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument("--model_save_name", required=True, type=str, help='model save log name')
    
    args = parser.parse_args()
    return args


class AMSTModels:
    """AMST双模型管理器"""
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        # 创建Alternative模型（sum融合）
        args_alt = argparse.Namespace(**vars(args))
        args_alt.fusion_method = 'sum'
        self.alt_model = self._create_model(args_alt)
        
        # 创建Joint模型（concat融合）
        args_joint = argparse.Namespace(**vars(args))
        args_joint.fusion_method = 'concat'
        self.joint_model = self._create_model(args_joint)
        
        self.modal_names = ast.literal_eval(args.model_name)
        self.num_modalities = len(self.modal_names)
        
        # 设置skip_factor映射
        self.skip_factors = {
            'Audio': args.skip_factor_audio,
            'Visual': args.skip_factor_visual,
            'Text': args.skip_factor_text,
            'Image': args.skip_factor_visual,  # Image使用Visual的skip_factor
        }
        
        self.skip_history = {name: [] for name in self.modal_names}
        
    def _create_model(self, args):
        if args.model_name == '["Visual","Audio"]' or args.model_name == '["Image","Text"]':
            model = VA_Classifier(args)
        else:
            model = TVA_Classifier(args)
        model.to(self.device)
        return model
    
    def should_skip_modality(self, modality_name, epoch):
        """判断当前epoch是否应该跳过该模态"""
        skip_factor = self.skip_factors.get(modality_name, 1)
        should_skip = (epoch % skip_factor != 0)
        if should_skip:
            self.skip_history[modality_name].append(epoch)
        return should_skip
    
    def print_skip_summary(self):
        """打印跳过统计信息"""
        print("\n" + "="*50)
        print("AMST Skip Summary:")
        print("="*50)
        for name in self.modal_names:
            skip_list = self.skip_history[name]
            skip_count = len(skip_list)
            skip_factor = self.skip_factors[name]
            print(f"{name}: skip_factor={skip_factor}, skipped {skip_count} times")
            if skip_count > 0 and skip_count <= 10:
                print(f"  Skipped epochs: {skip_list}")
        print("="*50 + "\n")


def train_epoch_amst(args, epoch, amst_models, device, dataloader, 
                     alt_optimizers, joint_optimizers, schedulers):
    """AMST训练函数"""
    epoch_start_time = time.time()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    amst_models.alt_model.train()
    amst_models.joint_model.train()
    
    modal_names = amst_models.modal_names
    num_modalities = amst_models.num_modalities
    
    # 性能指标
    metrics = {
        'alt': {'acc': [], 'f1': []},
        'joint': {'acc': [], 'f1': []},
        'final': {'acc': [], 'f1': []},
    }
    
    for name in modal_names:
        metrics[f'alt_{name}'] = {'acc': [], 'f1': []}
        metrics[f'joint_{name}'] = {'acc': [], 'f1': []}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs} [AMST Training]")
    
    for step, data_packet in enumerate(pbar):
        # 准备数据
        if modal_names == ["Visual", "Audio"]:
            spec, image, label = data_packet[0], data_packet[1], data_packet[2]
            spec, image, label = spec.to(device), image.to(device), label.to(device)
            if args.dataset == 'CREMAD':
                data_mini_packet = (image.float(), spec.float())
            else:
                data_mini_packet = (image.float(), spec.unsqueeze(1).float())
        elif modal_names == ["Image", "Text"]:
            token, padding_mask, image, label, _ = data_packet
            token = token.to(device)
            padding_mask = padding_mask.to(device)
            image, label = image.to(device), label.to(device)
            data_mini_packet = (token, padding_mask, image)
        elif modal_names == ["Text", "Visual", "Audio"]:
            token, padding_mask, image, spec, label, _ = data_packet
            token, padding_mask = token.to(device), padding_mask.to(device)
            image, spec, label = image.to(device), spec.to(device), label.to(device)
            data_mini_packet = (token, padding_mask, image.float(), spec.unsqueeze(1).float())
        else:
            raise NotImplementedError(f"Unsupported modal combination: {modal_names}")
        
        # ============= 1. Alternative模型训练 (单模态逐个训练) =============
        for modal_idx, modality_name in enumerate(modal_names):
            if amst_models.should_skip_modality(modality_name, epoch):
                continue  # 跳过该模态
            
            # 构造单模态输入（其他模态设为None）
            if modal_names == ["Visual", "Audio"]:
                visual_input = image.float() if modality_name == "Visual" else None
                if args.dataset == 'CREMAD':
                    audio_input = spec.float() if modality_name == "Audio" else None
                else:
                    audio_input = spec.unsqueeze(1).float() if modality_name == "Audio" else None
                data_single_modal = (visual_input, audio_input)
                
            elif modal_names == ["Image", "Text"]:
                if modality_name == "Text":
                    token_input = token
                    padding_mask_input = padding_mask
                else:
                    token_input = None
                    padding_mask_input = None
                image_input = image if modality_name == "Image" else None
                data_single_modal = (token_input, padding_mask_input, image_input)
                
            elif modal_names == ["Text", "Visual", "Audio"]:
                if modality_name == "Text":
                    token_input = token
                    padding_mask_input = padding_mask
                else:
                    token_input = None
                    padding_mask_input = None
                visual_input = image.float() if modality_name == "Visual" else None
                audio_input = spec.unsqueeze(1).float() if modality_name == "Audio" else None
                data_single_modal = (token_input, padding_mask_input, visual_input, audio_input)
            else:
                raise NotImplementedError(f"Unsupported modal combination: {modal_names}")
            
            # 清零梯度
            for opt in alt_optimizers.values():
                if opt is not None:
                    opt.zero_grad()
            
            # 前向传播（只使用一个模态）
            alt_outputs = amst_models.alt_model(data_single_modal, epoch=epoch, 
                                               labels=label, epoch_index=step)
            
            # 计算损失（使用融合输出）
            alt_fusion_logits = alt_outputs[0]
            
            # 使用融合输出计算损失（AMST原论文方法）
            loss_single = criterion(alt_fusion_logits, label).mean()
            
            # 反向传播
            loss_single.backward()
            nn.utils.clip_grad_norm_(amst_models.alt_model.parameters(), max_norm=40, norm_type=2)
            
            # 更新参数
            alt_optimizers['encoders'].step()
            if alt_optimizers['fusion'] is not None:
                alt_optimizers['fusion'].step()
        
        # ============= 2. Joint模型训练 (多模态联合训练) =============
        # 根据skip情况重新构造输入（跳过的模态设为None）
        if modal_names == ["Visual", "Audio"]:
            # 如果需要跳过，将对应模态设为None
            visual_input = None if amst_models.should_skip_modality("Visual", epoch) else image.float()
            if args.dataset == 'CREMAD':
                audio_input = None if amst_models.should_skip_modality("Audio", epoch) else spec.float()
            else:
                audio_input = None if amst_models.should_skip_modality("Audio", epoch) else spec.unsqueeze(1).float()
            data_mini_packet_joint = (visual_input, audio_input)
            
        elif modal_names == ["Image", "Text"]:
            # 如果需要跳过，将对应模态设为None
            if amst_models.should_skip_modality("Text", epoch):
                token_input = None
                padding_mask_input = None
            else:
                token_input = token
                padding_mask_input = padding_mask
            
            image_input = None if amst_models.should_skip_modality("Image", epoch) else image
            data_mini_packet_joint = (token_input, padding_mask_input, image_input)
                
        elif modal_names == ["Text", "Visual", "Audio"]:
            # 如果需要跳过，将对应模态设为None
            if amst_models.should_skip_modality("Text", epoch):
                token_input = None
                padding_mask_input = None
            else:
                token_input = token
                padding_mask_input = padding_mask
            
            visual_input = None if amst_models.should_skip_modality("Visual", epoch) else image.float()
            audio_input = None if amst_models.should_skip_modality("Audio", epoch) else spec.unsqueeze(1).float()
            data_mini_packet_joint = (token_input, padding_mask_input, visual_input, audio_input)
        else:
            raise NotImplementedError(f"Unsupported modal combination: {modal_names}")
        
        # 检查是否所有模态都被跳过
        all_skipped = all(amst_models.should_skip_modality(m, epoch) for m in modal_names)
        
        # 如果至少有一个模态未被跳过，则进行联合训练
        if not all_skipped:
            for opt in joint_optimizers.values():
                if opt is not None:
                    opt.zero_grad()
            
            joint_outputs = amst_models.joint_model(data_mini_packet_joint, epoch=epoch,
                                                   labels=label, epoch_index=step)
            
            joint_fusion_logits = joint_outputs[0]
            joint_modal_logits = list(joint_outputs[1:1 + num_modalities])
            
            # 联合损失（只计算未跳过模态的损失）
            loss_joint = criterion(joint_fusion_logits, label).mean()
            
            loss_joint.backward()
            nn.utils.clip_grad_norm_(amst_models.joint_model.parameters(), max_norm=40, norm_type=2)
            
            joint_optimizers['encoders'].step()
            if joint_optimizers['fusion'] is not None:
                joint_optimizers['fusion'].step()
        
        # ============= 3. Helper辅助分类器训练 =============
        if args.use_helper and hasattr(amst_models.alt_model, 'helper'):
            # Alternative模型的helper
            alt_outputs_helper = amst_models.alt_model(data_mini_packet, epoch=epoch,
                                                      labels=label, epoch_index=step)
            if len(alt_outputs_helper) > 1 + num_modalities:
                alt_optimizers['helper'].zero_grad()
                helper_losses = []
                for modal_idx in range(num_modalities):
                    if not amst_models.should_skip_modality(modal_names[modal_idx], epoch):
                        # Helper输出通常在最后
                        helper_logits = alt_outputs_helper[1 + num_modalities + modal_idx]
                        helper_loss = criterion(helper_logits, label).mean()
                        helper_losses.append(helper_loss)
                
                if helper_losses:
                    total_helper_loss = sum(helper_losses) * args.helper_weight
                    total_helper_loss.backward()
                    alt_optimizers['helper'].step()
        
        # ============= 4. 计算性能指标（用于显示，不计算梯度） =============
        with torch.no_grad():
            # Alternative模型预测
            alt_outputs_eval = amst_models.alt_model(data_mini_packet)
            alt_fusion_pred = alt_outputs_eval[0]
            alt_modal_preds = list(alt_outputs_eval[1:1 + num_modalities])
            
            # Joint模型预测
            joint_outputs_eval = amst_models.joint_model(data_mini_packet)
            joint_fusion_pred = joint_outputs_eval[0]
            joint_modal_preds = list(joint_outputs_eval[1:1 + num_modalities])
            
            # AMST最终预测：加权组合
            final_pred = (alt_fusion_pred * num_modalities + joint_fusion_pred) / (num_modalities + 1)
            
            # 计算准确率
            acc_alt, f1_alt = calculate_metrics(alt_fusion_pred, label)
            acc_joint, f1_joint = calculate_metrics(joint_fusion_pred, label)
            acc_final, f1_final = calculate_metrics(final_pred, label)
            
            metrics['alt']['acc'].append(acc_alt)
            metrics['alt']['f1'].append(f1_alt)
            metrics['joint']['acc'].append(acc_joint)
            metrics['joint']['f1'].append(f1_joint)
            metrics['final']['acc'].append(acc_final)
            metrics['final']['f1'].append(f1_final)
            
            # 单模态指标
            for idx, name in enumerate(modal_names):
                acc_a, f1_a = calculate_metrics(alt_modal_preds[idx], label)
                acc_j, f1_j = calculate_metrics(joint_modal_preds[idx], label)
                metrics[f'alt_{name}']['acc'].append(acc_a)
                metrics[f'alt_{name}']['f1'].append(f1_a)
                metrics[f'joint_{name}']['acc'].append(acc_j)
                metrics[f'joint_{name}']['f1'].append(f1_j)
        
        # 更新进度条
        postfix = {
            'Final_Acc': f'{acc_final:.4f}',
            'Alt_Acc': f'{acc_alt:.4f}',
            'Joint_Acc': f'{acc_joint:.4f}',
        }
        pbar.set_postfix(postfix)
    
    # 学习率衰减
    if schedulers is not None:
        for sch in schedulers:
            sch.step()
    
    # 计算epoch平均指标
    avg_metrics = {}
    for key, value in metrics.items():
        if 'acc' in value:
            avg_acc = sum(value['acc']) / len(value['acc']) if value['acc'] else 0.0
            avg_f1 = sum(value['f1']) / len(value['f1']) if value['f1'] else 0.0
            avg_metrics[key] = (avg_acc, avg_f1)
    
    epoch_time = time.time() - epoch_start_time
    
    # 打印总结
    print(f"\nTrain Epoch {epoch} Summary:")
    print(f"  Final   -> Accuracy: {avg_metrics['final'][0]:.4f}, F1-Score: {avg_metrics['final'][1]:.4f}")
    print(f"  Alt     -> Accuracy: {avg_metrics['alt'][0]:.4f}, F1-Score: {avg_metrics['alt'][1]:.4f}")
    print(f"  Joint   -> Accuracy: {avg_metrics['joint'][0]:.4f}, F1-Score: {avg_metrics['joint'][1]:.4f}")
    print(f"  Epoch Time: {epoch_time:.2f}s ({epoch_time/60:.2f}min)")
    
    return avg_metrics, epoch_time


def valid_amst(args, amst_models, device, dataloader):
    """AMST验证函数"""
    with torch.no_grad():
        amst_models.alt_model.eval()
        amst_models.joint_model.eval()
        
        modal_names = amst_models.modal_names
        num_modalities = amst_models.num_modalities
        
        all_labels = []
        alt_preds, joint_preds, final_preds = [], [], []
        alt_modal_preds = [[] for _ in range(num_modalities)]
        joint_modal_preds = [[] for _ in range(num_modalities)]
        
        pbar = tqdm(dataloader, desc="Validating")
        for step, data_packet in enumerate(pbar):
            # 准备数据
            if modal_names == ["Visual", "Audio"]:
                spec, image, label = data_packet[0], data_packet[1], data_packet[2]
                spec, image, label = spec.to(device), image.to(device), label.to(device)
                if args.dataset == 'CREMAD':
                    data_mini_packet = (image.float(), spec.float())
                else:
                    data_mini_packet = (image.float(), spec.unsqueeze(1).float())
            elif modal_names == ["Image", "Text"]:
                token, padding_mask, image, label, _ = data_packet
                token, padding_mask = token.to(device), padding_mask.to(device)
                image, label = image.to(device), label.to(device)
                data_mini_packet = (token, padding_mask, image)
            elif modal_names == ["Text", "Visual", "Audio"]:
                token, padding_mask, image, spec, label, _ = data_packet
                token, padding_mask = token.to(device), padding_mask.to(device)
                image, spec, label = image.to(device), spec.to(device), label.to(device)
                data_mini_packet = (token, padding_mask, image.float(), spec.unsqueeze(1).float())
            else:
                raise NotImplementedError(f"Unsupported modal combination: {modal_names}")
            
            # Alternative模型预测
            alt_outputs = amst_models.alt_model(data_mini_packet)
            alt_fusion = alt_outputs[0]
            alt_modals = list(alt_outputs[1:1 + num_modalities])
            
            # Joint模型预测
            joint_outputs = amst_models.joint_model(data_mini_packet)
            joint_fusion = joint_outputs[0]
            joint_modals = list(joint_outputs[1:1 + num_modalities])
            
            # AMST最终预测
            final_fusion = (alt_fusion * num_modalities + joint_fusion) / (num_modalities + 1)
            
            all_labels.append(label)
            alt_preds.append(alt_fusion)
            joint_preds.append(joint_fusion)
            final_preds.append(final_fusion)
            
            for idx in range(num_modalities):
                alt_modal_preds[idx].append(alt_modals[idx])
                joint_modal_preds[idx].append(joint_modals[idx])
        
        # 合并所有batch
        all_labels = torch.cat(all_labels, dim=0)
        alt_preds = torch.cat(alt_preds, dim=0)
        joint_preds = torch.cat(joint_preds, dim=0)
        final_preds = torch.cat(final_preds, dim=0)
        alt_modal_preds = [torch.cat(preds, dim=0) for preds in alt_modal_preds]
        joint_modal_preds = [torch.cat(preds, dim=0) for preds in joint_modal_preds]
        
        # 计算指标
        acc_final, f1_final = calculate_metrics(final_preds, all_labels)
        acc_alt, f1_alt = calculate_metrics(alt_preds, all_labels)
        acc_joint, f1_joint = calculate_metrics(joint_preds, all_labels)
        
        modal_metrics = {}
        for idx, name in enumerate(modal_names):
            acc_a, f1_a = calculate_metrics(alt_modal_preds[idx], all_labels)
            acc_j, f1_j = calculate_metrics(joint_modal_preds[idx], all_labels)
            modal_metrics[f'alt_{name}'] = (acc_a, f1_a)
            modal_metrics[f'joint_{name}'] = (acc_j, f1_j)
        
        print(f"\nValidation Summary:")
        print(f"  Final   -> Accuracy: {acc_final:.4f}, F1-Score: {f1_final:.4f}")
        print(f"  Alt     -> Accuracy: {acc_alt:.4f}, F1-Score: {f1_alt:.4f}")
        print(f"  Joint   -> Accuracy: {acc_joint:.4f}, F1-Score: {f1_joint:.4f}")
        
        for name in modal_names:
            acc_a, f1_a = modal_metrics[f'alt_{name}']
            acc_j, f1_j = modal_metrics[f'joint_{name}']
            print(f"  {name}: Alt({acc_a:.4f}), Joint({acc_j:.4f})")
    
    return (acc_final, f1_final), (acc_alt, f1_alt), (acc_joint, f1_joint), modal_metrics


def main():
    args = get_arguments()
    print(args)
    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ==================数据集加载=====================================
    if args.dataset == 'CREMAD':
        args.num_classes = 6
        train_dataset = AV_CD_Dataset(mode='train')
        test_dataset = AV_CD_Dataset(mode='test')
    elif args.dataset == 'KineticSound':
        args.num_classes = 34
        train_dataset = KSDataset(mode='train', args=args)
        test_dataset = KSDataset(mode='test', args=args)
    elif args.dataset == 'AVE':
        args.num_classes = 28
        train_dataset = AVEDataset(mode='train', args=args)
        test_dataset = AVEDataset(mode='test', args=args)
    elif args.dataset == 'Food101':
        args.num_classes = 101
        train_dataset = M3AEDataset(args, mode='train')
        test_dataset = M3AEDataset(args, mode='test')
    elif args.dataset == 'MVSA':
        args.num_classes = 3
        train_dataset = M3AEDataset(args, mode='train')
        test_dataset = M3AEDataset(args, mode='test')
    elif args.dataset == 'IEMOCAP3':
        args.num_classes = 5
        train_dataset = TVADataset(mode='train', args=args, pick_num=3)
        test_dataset = TVADataset(mode='test', args=args, pick_num=3)
    else:
        raise NotImplementedError(f'Incorrect dataset name {args.dataset}!')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                 num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
    
    # ==================AMST模型初始化=====================================
    print("\n" + "="*50)
    print("Initializing AMST Models...")
    print("="*50)
    
    amst_models = AMSTModels(args, device)
    
    print(f"\nAlternative Model (Sum Fusion):")
    print_model_params(amst_models.alt_model)
    print(f"\nJoint Model (Concat Fusion):")
    print_model_params(amst_models.joint_model)
    
    # ==================优化器设置=====================================
    def create_optimizers(model, args):
        encoder_params, fusion_params, helper_params = [], [], []
        
        for name, param in model.named_parameters():
            if 'encoder' in name or 'm1' in name or 'm2' in name or 'm3' in name:
                encoder_params.append(param)
            elif 'helper' in name:
                helper_params.append(param)
            else:
                print(f"name is {name}")
                fusion_params.append(param)
        
        optimizers = {}
        if args.optimizer == "Adamw":
            optimizers['encoders'] = torch.optim.AdamW(encoder_params, lr=args.learning_rate_fusion,
                                                      betas=(0.9, 0.999), weight_decay=args.weight_decay)
            if fusion_params != []:
                optimizers['fusion'] = torch.optim.AdamW(fusion_params, lr=args.learning_rate_fusion,
                                                    betas=(0.9, 0.999), weight_decay=args.weight_decay)
            else:
                optimizers['fusion'] = None
            if args.use_helper and helper_params:
                optimizers['helper'] = torch.optim.AdamW(helper_params, lr=args.learning_rate_fusion,
                                                        betas=(0.9, 0.999), weight_decay=args.weight_decay)
        
        return optimizers
    
    alt_optimizers = create_optimizers(amst_models.alt_model, args)
    joint_optimizers = create_optimizers(amst_models.joint_model, args)
    
    # 学习率调度器
    schedulers = []
    for opt in list(alt_optimizers.values()) + list(joint_optimizers.values()):
        if opt is not None:
            sch = torch.optim.lr_scheduler.StepLR(opt, args.lr_decay_step, args.lr_decay_ratio)
            schedulers.append(sch)
    
    # ==================日志设置=====================================
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    
    log_path = os.path.join(args.ckpt_path, args.dataset + '_amst.csv')
    with open(log_path, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Val_Final_Acc', 'Val_Final_F1', 'Val_Alt_Acc', 'Val_Alt_F1',
                        'Val_Joint_Acc', 'Val_Joint_F1'])
    
    # ==================训练和验证=====================================
    if args.train:
        best_acc = 0.0
        save_path = None
        epoch_times = []
        
        print("\n" + "="*50)
        print("Starting AMST Training...")
        print(f"Skip Factors: Audio={args.skip_factor_audio}, Visual={args.skip_factor_visual}, Text={args.skip_factor_text}")
        print("="*50 + "\n")
        
        for epoch in range(args.current_epoch, args.epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*50}")
            
            # 训练
            train_metrics, epoch_time = train_epoch_amst(
                args=args,
                epoch=epoch,
                amst_models=amst_models,
                device=device,
                dataloader=train_dataloader,
                alt_optimizers=alt_optimizers,
                joint_optimizers=joint_optimizers,
                schedulers=schedulers
            )
            epoch_times.append(epoch_time)
            
            # 验证
            val_final_metrics, val_alt_metrics, val_joint_metrics, val_modal_metrics = \
                valid_amst(args, amst_models, device, test_dataloader)
            
            # Tensorboard
            if args.use_tensorboard:
                writer = SummaryWriter(log_dir=args.tensorboard_path)
                writer.add_scalar('Val/Final_Accuracy', val_final_metrics[0], epoch)
                writer.add_scalar('Val/Alt_Accuracy', val_alt_metrics[0], epoch)
                writer.add_scalar('Val/Joint_Accuracy', val_joint_metrics[0], epoch)
                writer.close()
            
            # 保存最佳模型
            is_best = val_final_metrics[0] > best_acc
            best_acc = max(val_final_metrics[0], best_acc)
            print(f"\nBest Val Acc: {best_acc:.4f}")
            
            if is_best:
                model_name = f"{args.model_save_name}_amst_best_acc{best_acc:.4f}_epoch{epoch}.pth"
                save_path = os.path.join(args.ckpt_path, model_name)
                
                torch.save({
                    'epoch': epoch,
                    'alt_model_state_dict': amst_models.alt_model.state_dict(),
                    'joint_model_state_dict': amst_models.joint_model.state_dict(),
                    'best_acc': best_acc,
                    'args': args
                }, save_path)
                
                print(f"✓ Best model saved: {save_path}")
            
            # 记录日志
            with open(log_path, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch, val_final_metrics[0], val_final_metrics[1],
                               val_alt_metrics[0], val_alt_metrics[1],
                               val_joint_metrics[0], val_joint_metrics[1]])
        
        # 打印跳过统计
        amst_models.print_skip_summary()
        
        # 输出训练时间统计
        print(f"\n{'='*50}")
        print(f"Training Time Statistics")
        print(f"{'='*50}")
        if epoch_times:
            total_time = sum(epoch_times)
            avg_time = total_time / len(epoch_times)
            print(f"Total Training Time: {total_time:.2f}s ({total_time/60:.2f}min / {total_time/3600:.2f}h)")
            print(f"Average Epoch Time: {avg_time:.2f}s ({avg_time/60:.2f}min)")
        print(f"{'='*50}\n")
        
        # 写入结果日志
        results_dir = "Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        log_file = os.path.join(results_dir, f"results-AMST-{args.dataset}.log")
        with open(log_file, "a") as f:
            f.write(f"==================== {datetime.datetime.now()} ===================\n\n")
            f.write(f"========================={args.model_save_name}==================================\n")
            f.write(f"val_acc: {best_acc}\n")
            f.write(f"val_final_metrics: {val_final_metrics}\n")
            f.write(f"val_alt_metrics: {val_alt_metrics}\n")
            f.write(f"val_joint_metrics: {val_joint_metrics}\n")
            f.write(f"best model save as {save_path}\n")
            if epoch_times:
                total_time = sum(epoch_times)
                avg_time = total_time / len(epoch_times)
                f.write(f"total_training_time: {total_time:.2f}s ({total_time/60:.2f}min / {total_time/3600:.2f}h)\n")
                f.write(f"avg_epoch_time: {avg_time:.2f}s ({avg_time/60:.2f}min)\n")
            f.write(f"args: {args}\n\n")
        
        print(f"Best Val Acc: {best_acc:.4f}")
        print(f"Model saved to: {save_path}")
        print(f"Results logged to: {log_file}")


if __name__ == '__main__':
    main()
