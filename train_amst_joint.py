"""
AMST Joint Training Implementation
Based on amst_joint.py and train_amst.py frameworks
Uses single concat fusion model with modality skip mechanism
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
import pickle as pickle

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
    parser.add_argument('--dataset', default='CREMAD', type=str, 
                        help='KineticSound, CREMAD, AVE, MVSA, Food101, IEMOCAP3')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--train', action='store_true', help='turn on train mode')
    parser.add_argument('--use_video_frames', default=3, type=int, help='use how many frames for train')
    
    # 学习率定义
    parser.add_argument('--learning_rate_visual', default=1e-3, type=float, 
                        help='Learning rate for visual encoder')
    parser.add_argument('--learning_rate_audio', default=1e-3, type=float, 
                        help='Learning rate for audio encoder')
    parser.add_argument('--learning_rate_text', default=1e-5, type=float, 
                        help='Learning rate for text encoder')
    parser.add_argument('--learning_rate_image', default=1e-3, type=float, 
                        help='Learning rate for image encoder')
    parser.add_argument('--learning_rate_fusion', default=1e-3, type=float, 
                        help='Learning rate for fusion model')
    parser.add_argument('--optimizer', default='Adamw', type=str, 
                        help='Optimizer to use (SGD, Adam, Adamw)')
    parser.add_argument('--lr_decay_step', default=30, type=int)
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    
    # 基础设置
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU ids')
    parser.add_argument('--current_epoch', type=int, default=1, help="Start train epoch number")
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    
    # 模型设置 - AMST Joint only uses concat fusion
    parser.add_argument('--fusion_method', default='concat', type=str, choices=['concat'], 
                        help='Fusion method (AMST Joint only supports concat)')
    parser.add_argument('--model_name', default='["Visual","Audio"]', type=str,
                        choices=['["Visual","Audio"]', '["Image","Text"]', '["Text","Visual","Audio"]'])
    parser.add_argument('--modality', default='full', type=str,
                        choices=['full', 'audio', 'visual', 'text', 'image'],
                        help='modality to use')
    parser.add_argument('--unified_dim', default=512, type=int, 
                        help='Unified feature dimension after encoders')
    parser.add_argument('--m1_token_len', default=1, type=int, help='Modality 1 token length')
    parser.add_argument('--m2_token_len', default=1, type=int, help='Modality 2 token length')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes')
    
    # AMST Skip Factor Parameters
    parser.add_argument('--skip_visual', default=1, type=int, 
                        help='Skip factor for visual modality (1=no skip)')
    parser.add_argument('--skip_audio', default=1, type=int, 
                        help='Skip factor for audio modality (1=no skip)')
    parser.add_argument('--skip_text', default=1, type=int, 
                        help='Skip factor for text modality (1=no skip)')
    parser.add_argument('--skip_image', default=1, type=int, 
                        help='Skip factor for image modality (1=no skip)')
    parser.add_argument('--alpha', default=1.0, type=float, 
                        help='Weight for single modality losses')
    
    # AME相关参数（保持兼容）
    parser.add_argument('--MaskType', default='None', type=str, choices=['None', 'AME'], 
                        help='Type of masking strategy')
    parser.add_argument('--ame_gama', default=0.1, type=float, help='Gamma parameter for AME module')
    parser.add_argument('--ame_beta', default=0.7, type=float, help='Beta parameter for AME module')
    parser.add_argument('--ame_gap', default=2, type=int, help='restoration gap')
    parser.add_argument('--ame_gap_start', default=1, type=int, help='restoration gap start')
    parser.add_argument('--ame_temperature', default=0.2, type=float, 
                        help='Temperature parameter for AME module')
    parser.add_argument('--warmup_epoch', default=0, type=int, 
                        help='Number of warmup epochs before applying AME')
    parser.add_argument('--Use_MACE', default=True, type=bool, help="是否使用MACE loss")
    
    # tensorboard相关参数
    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')
    
    # 保存模型的相关参数
    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument("--model_save_name", required=True, type=str, help='model save log name')
    
    args = parser.parse_args()
    return args


def should_skip_modality(modality_name, epoch, skip_factors):
    """判断当前epoch是否跳过某个模态的训练"""
    skip_factor = skip_factors.get(modality_name, 1)
    return epoch % skip_factor != 0


def train_epoch_amst_joint(args, epoch, model, device, dataloader, optimizer_map, 
                           scheduler_map, skip_factors):
    """
    AMST Joint Training: 使用concat融合，根据skip_factor决定是否跳过某些模态
    参考amst_joint.py的joint_train逻辑
    """
    epoch_start_time = time.time()
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.train()

    modal_names = ast.literal_eval(args.model_name)
    num_modalities = len(modal_names)

    all_acc_fusion, all_f1_fusion = [], []
    modal_acc_lists = [[] for _ in range(num_modalities)]
    modal_f1_lists = [[] for _ in range(num_modalities)]
    modal_loss_sums = [0.0 for _ in range(num_modalities)]
    
    # 记录本epoch跳过的模态
    skipped_modalities = []
    for modal_name in modal_names:
        if should_skip_modality(modal_name, epoch, skip_factors):
            skipped_modalities.append(modal_name)
    
    if skipped_modalities:
        print(f"Epoch {epoch}: Skipping modalities: {skipped_modalities}")
    
    # 检查是否所有模态都被跳过
    all_skipped = all(should_skip_modality(m, epoch, skip_factors) for m in modal_names)
    if all_skipped:
        print(f"WARNING: All modalities are skipped at epoch {epoch}. No training will occur.")
        return {
            'acc_fusion': 0.0,
            'f1_fusion': 0.0,
            'loss_fusion': 0.0,
            'modal_metrics': {name: {'acc': 0.0, 'f1': 0.0, 'loss': 0.0} for name in modal_names},
            'epoch_time': 0.0
        }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs} [AMST Joint Training]")
    
    for step, data_packet in enumerate(pbar):
        # 清零梯度
        for opt in optimizer_map.values():
            if opt is not None:
                opt.zero_grad()

        # # 准备数据输入（跳过的模态设为全零）
        # if modal_names == ["Visual", "Audio"]:
        #     spec, image, label = data_packet[0], data_packet[1], data_packet[2]
        #     spec, image, label = spec.to(device), image.to(device), label.to(device)
            
        #     # 如果需要跳过，将对应模态设为全零
        #     visual_input = torch.zeros_like(image).float() if should_skip_modality("Visual", epoch, skip_factors) else image.float()
        #     if args.dataset == 'CREMAD':
        #         audio_input = torch.zeros_like(spec).float() if should_skip_modality("Audio", epoch, skip_factors) else spec.float()
        #     else:
        #         audio_input = torch.zeros_like(spec.unsqueeze(1)).float() if should_skip_modality("Audio", epoch, skip_factors) else spec.unsqueeze(1).float()
            
        #     data_mini_packet = (visual_input, audio_input)
            
        # elif modal_names == ["Image", "Text"]:
        #     token, padding_mask, image, label, _ = data_packet
        #     token, padding_mask = token.to(device), padding_mask.to(device)
        #     image, label = image.to(device), label.to(device)
            
        #     # 如果需要跳过，将对应模态设为全零
        #     if should_skip_modality("Text", epoch, skip_factors):
        #         token_input = torch.zeros_like(token)
        #         padding_mask_input = torch.zeros_like(padding_mask)
        #     else:
        #         token_input = token
        #         padding_mask_input = padding_mask
            
        #     image_input = torch.zeros_like(image) if should_skip_modality("Image", epoch, skip_factors) else image
            
        #     data_mini_packet = (token_input, padding_mask_input, image_input)
                
        # elif modal_names == ["Text", "Visual", "Audio"]:
        #     token, padding_mask, image, spec, label, _ = data_packet
        #     token, padding_mask = token.to(device), padding_mask.to(device)
        #     image, spec, label = image.to(device), spec.to(device), label.to(device)
            
        #     # 如果需要跳过，将对应模态设为全零
        #     if should_skip_modality("Text", epoch, skip_factors):
        #         token_input = torch.zeros_like(token)
        #         padding_mask_input = torch.zeros_like(padding_mask)
        #     else:
        #         token_input = token
        #         padding_mask_input = padding_mask
            
        #     visual_input = torch.zeros_like(image).float() if should_skip_modality("Visual", epoch, skip_factors) else image.float()
        #     audio_input = torch.zeros_like(spec.unsqueeze(1)).float() if should_skip_modality("Audio", epoch, skip_factors) else spec.unsqueeze(1).float()
            
        #     data_mini_packet = (token_input, padding_mask_input, visual_input, audio_input)
        # else:
        #     raise NotImplementedError(f"Unsupported modal combination: {modal_names}")
        # 准备数据输入（跳过的模态设为None）
        if modal_names == ["Visual", "Audio"]:
            spec, image, label = data_packet[0], data_packet[1], data_packet[2]
            spec, image, label = spec.to(device), image.to(device), label.to(device)
            
            # 如果需要跳过，将对应模态设为None
            visual_input = None if should_skip_modality("Visual", epoch, skip_factors) else image.float()
            if args.dataset == 'CREMAD':
                audio_input = None if should_skip_modality("Audio", epoch, skip_factors) else spec.float()
            else:
                audio_input = None if should_skip_modality("Audio", epoch, skip_factors) else spec.unsqueeze(1).float()
            
            data_mini_packet = (visual_input, audio_input)
            
        elif modal_names == ["Image", "Text"]:
            token, padding_mask, image, label, _ = data_packet
            token, padding_mask = token.to(device), padding_mask.to(device)
            image, label = image.to(device), label.to(device)
            
            # 如果需要跳过，将对应模态设为None
            if should_skip_modality("Text", epoch, skip_factors):
                token_input = None
                padding_mask_input = None
            else:
                token_input = token
                padding_mask_input = padding_mask
            
            image_input = None if should_skip_modality("Image", epoch, skip_factors) else image
            
            data_mini_packet = (token_input, padding_mask_input, image_input)
                
        elif modal_names == ["Text", "Visual", "Audio"]:
            token, padding_mask, image, spec, label, _ = data_packet
            token, padding_mask = token.to(device), padding_mask.to(device)
            image, spec, label = image.to(device), spec.to(device), label.to(device)
            
            # 如果需要跳过，将对应模态设为None
            if should_skip_modality("Text", epoch, skip_factors):
                token_input = None
                padding_mask_input = None
            else:
                token_input = token
                padding_mask_input = padding_mask
            
            visual_input = None if should_skip_modality("Visual", epoch, skip_factors) else image.float()
            audio_input = None if should_skip_modality("Audio", epoch, skip_factors) else spec.unsqueeze(1).float()
            
            data_mini_packet = (token_input, padding_mask_input, visual_input, audio_input)
        else:
            raise NotImplementedError(f"Unsupported modal combination: {modal_names}")

        # 前向传播
        outputs = model(data_mini_packet, epoch=epoch, labels=label, epoch_index=step)
        
        fusion_logits = outputs[0]
        modal_logits = list(outputs[1:1 + num_modalities])
        extra = list(outputs[1 + num_modalities:])

        # 提取masks（用于AME）
        masks = []
        if extra:
            if len(extra) == num_modalities:
                masks = extra
            elif len(extra) == num_modalities + 1:
                masks = extra[1:]
        masks = [m.float() if isinstance(m, torch.Tensor) else None for m in masks]

        # 计算融合结果的准确率和F1
        acc_fusion, f1_fusion = calculate_metrics(fusion_logits, label)
        all_acc_fusion.append(acc_fusion)
        all_f1_fusion.append(f1_fusion)

        # 计算每个模态的损失
        modal_losses_each = [criterion(logit, label) for logit in modal_logits]
        weighted_modal_losses = []
        modal_loss_values = []
        
        for idx, (loss_each, mask) in enumerate(zip(modal_losses_each, masks + [None] * (num_modalities - len(masks)))):
            # 如果该模态被跳过，损失设为0
            if should_skip_modality(modal_names[idx], epoch, skip_factors):
                weighted = torch.tensor(0.0, device=device)
            else:
                if mask is not None and args.Use_MACE:
                    weight = mask.view(loss_each.size(0)).clamp_min(0)
                    weighted = (loss_each * weight).sum() / weight.sum().clamp_min(1e-6)
                else:
                    weighted = loss_each.mean()
            
            weighted_modal_losses.append(weighted)
            modal_loss_values.append(weighted.detach().item())

        # 计算每个模态的指标（只对未跳过的模态计算）
        for idx, logits in enumerate(modal_logits):
            if not should_skip_modality(modal_names[idx], epoch, skip_factors):
                acc_i, f1_i = calculate_metrics(logits, label)
                modal_acc_lists[idx].append(acc_i)
                modal_f1_lists[idx].append(f1_i)
                if idx < len(modal_loss_values):
                    modal_loss_sums[idx] += modal_loss_values[idx]

        # 总损失：融合损失 + 各模态损失
        loss_fusion = criterion(fusion_logits, label).mean()
        loss_single = sum(weighted_modal_losses) * args.alpha
        loss = loss_fusion + loss_single

        # 反向传播
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)
        
        # 更新参数（只更新未跳过模态的编码器）
        for opt_name, opt in optimizer_map.items():
            if opt is not None:
                # 检查是否为某个模态的编码器优化器
                if opt_name.startswith('encoder_'):
                    modality = opt_name.replace('encoder_', '').capitalize()
                    if not should_skip_modality(modality, epoch, skip_factors):
                        opt.step()
                else:
                    # 融合层优化器总是更新
                    opt.step()

        # 更新进度条
        postfix = {
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{acc_fusion:.4f}',
            'F1': f'{f1_fusion:.4f}',
        }
        for idx, name in enumerate(modal_names):
            if not should_skip_modality(name, epoch, skip_factors) and modal_acc_lists[idx]:
                postfix[f'Acc_{name}'] = f'{modal_acc_lists[idx][-1]:.4f}'
                postfix[f'Loss_{name}'] = f'{modal_loss_values[idx]:.4f}'
            else:
                postfix[f'Acc_{name}'] = 'SKIP'
                postfix[f'Loss_{name}'] = 'SKIP'
        pbar.set_postfix(postfix)

    # 更新学习率调度器
    if scheduler_map is not None:
        for sch in scheduler_map.values():
            if sch is not None:
                sch.step()

    epoch_time = time.time() - epoch_start_time

    # 统计结果
    mean_acc_fusion = np.mean(all_acc_fusion) if all_acc_fusion else 0.0
    mean_f1_fusion = np.mean(all_f1_fusion) if all_f1_fusion else 0.0
    mean_loss_fusion = modal_loss_sums[0] / len(dataloader) if len(dataloader) > 0 else 0.0

    modal_metrics = {}
    for idx, name in enumerate(modal_names):
        if modal_acc_lists[idx]:  # 只统计未跳过的模态
            modal_metrics[name] = {
                'acc': np.mean(modal_acc_lists[idx]),
                'f1': np.mean(modal_f1_lists[idx]),
                'loss': modal_loss_sums[idx] / len(dataloader) if len(dataloader) > 0 else 0.0
            }
        else:
            modal_metrics[name] = {'acc': 0.0, 'f1': 0.0, 'loss': 0.0}

    return {
        'acc_fusion': mean_acc_fusion,
        'f1_fusion': mean_f1_fusion,
        'loss_fusion': mean_loss_fusion,
        'modal_metrics': modal_metrics,
        'epoch_time': epoch_time
    }


def valid_amst_joint(args, epoch, model, device, dataloader):
    """
    AMST Joint Validation: 使用所有模态进行验证
    """
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.eval()

    modal_names = ast.literal_eval(args.model_name)
    num_modalities = len(modal_names)

    all_acc_fusion, all_f1_fusion = [], []
    modal_acc_lists = [[] for _ in range(num_modalities)]
    modal_f1_lists = [[] for _ in range(num_modalities)]
    modal_loss_sums = [0.0 for _ in range(num_modalities)]

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs} [Validation]")
    
    with torch.no_grad():
        for step, data_packet in enumerate(pbar):
            # 准备数据输入（验证时使用所有模态）
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

            # 前向传播
            outputs = model(data_mini_packet, epoch=epoch, labels=label, epoch_index=step)
            
            fusion_logits = outputs[0]
            modal_logits = list(outputs[1:1 + num_modalities])
            extra = list(outputs[1 + num_modalities:])

            # 提取masks
            masks = []
            if extra:
                if len(extra) == num_modalities:
                    masks = extra
                elif len(extra) == num_modalities + 1:
                    masks = extra[1:]
            masks = [m.float() if isinstance(m, torch.Tensor) else None for m in masks]

            # 计算融合结果的准确率和F1
            acc_fusion, f1_fusion = calculate_metrics(fusion_logits, label)
            all_acc_fusion.append(acc_fusion)
            all_f1_fusion.append(f1_fusion)

            # 计算每个模态的损失
            modal_losses_each = [criterion(logit, label) for logit in modal_logits]
            modal_loss_values = []
            
            for idx, (loss_each, mask) in enumerate(zip(modal_losses_each, masks + [None] * (num_modalities - len(masks)))):
                if mask is not None and args.Use_MACE:
                    weight = mask.view(loss_each.size(0)).clamp_min(0)
                    weighted = (loss_each * weight).sum() / weight.sum().clamp_min(1e-6)
                else:
                    weighted = loss_each.mean()
                modal_loss_values.append(weighted.detach().item())

            # 计算每个模态的指标
            for idx, logits in enumerate(modal_logits):
                acc_i, f1_i = calculate_metrics(logits, label)
                modal_acc_lists[idx].append(acc_i)
                modal_f1_lists[idx].append(f1_i)
                if idx < len(modal_loss_values):
                    modal_loss_sums[idx] += modal_loss_values[idx]

            # 更新进度条
            postfix = {
                'Acc': f'{acc_fusion:.4f}',
                'F1': f'{f1_fusion:.4f}',
            }
            for idx, name in enumerate(modal_names):
                if modal_acc_lists[idx]:
                    postfix[f'Acc_{name}'] = f'{modal_acc_lists[idx][-1]:.4f}'
            pbar.set_postfix(postfix)

    # 统计结果
    mean_acc_fusion = np.mean(all_acc_fusion) if all_acc_fusion else 0.0
    mean_f1_fusion = np.mean(all_f1_fusion) if all_f1_fusion else 0.0
    mean_loss_fusion = modal_loss_sums[0] / len(dataloader) if len(dataloader) > 0 else 0.0

    modal_metrics = {}
    for idx, name in enumerate(modal_names):
        modal_metrics[name] = {
            'acc': np.mean(modal_acc_lists[idx]) if modal_acc_lists[idx] else 0.0,
            'f1': np.mean(modal_f1_lists[idx]) if modal_f1_lists[idx] else 0.0,
            'loss': modal_loss_sums[idx] / len(dataloader) if len(dataloader) > 0 else 0.0
        }

    return {
        'acc_fusion': mean_acc_fusion,
        'f1_fusion': mean_f1_fusion,
        'loss_fusion': mean_loss_fusion,
        'modal_metrics': modal_metrics
    }


def main():
    args = get_arguments()
    print(args)
    
    # 设置随机种子
    setup_seed(args.random_seed)
    
    # 设置设备
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 构建skip_factors字典
    skip_factors = {
        'Visual': args.skip_visual,
        'Audio': args.skip_audio,
        'Text': args.skip_text,
        'Image': args.skip_image,
    }
    
    modal_names = ast.literal_eval(args.model_name)
    print(f"Training with modalities: {modal_names}")
    print(f"Skip factors: {skip_factors}")
    
    # ==================数据集加载=====================================
    if args.dataset == 'CREMAD':
        args.num_classes = 6
        train_dataset = AV_CD_Dataset(mode='train')
        val_dataset = AV_CD_Dataset(mode='test')
    elif args.dataset == 'KineticSound':
        args.num_classes = 34
        train_dataset = KSDataset(mode='train', args=args)
        val_dataset = KSDataset(mode='test', args=args)
    elif args.dataset == 'AVE':
        args.num_classes = 28
        train_dataset = AVEDataset(mode='train', args=args)
        val_dataset = AVEDataset(mode='test', args=args)
    elif args.dataset == 'Food101':
        args.num_classes = 101
        train_dataset = M3AEDataset(args, mode='train')
        val_dataset = M3AEDataset(args, mode='test')
    elif args.dataset == 'MVSA':
        args.num_classes = 3
        train_dataset = M3AEDataset(args, mode='train')
        val_dataset = M3AEDataset(args, mode='test')
    elif args.dataset == 'IEMOCAP3':
        args.num_classes = 5
        train_dataset = TVADataset(mode='train', args=args, pick_num=3)
        val_dataset = TVADataset(mode='test', args=args, pick_num=3)
    else:
        raise NotImplementedError(f'Incorrect dataset name {args.dataset}!')
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # ==================模型初始化=====================================
    print("\n" + "="*50)
    print("Initializing AMST Joint Model (Concat Fusion)...")
    print("="*50)
    
    if len(modal_names) == 2:
        model = VA_Classifier(args).to(device)
    elif len(modal_names) == 3:
        model = TVA_Classifier(args).to(device)
    else:
        raise NotImplementedError(f"Unsupported number of modalities: {len(modal_names)}")
    
    # 初始化模型参数
    model.apply(weight_init)
    print_model_params(model)
    
    # ==================优化器设置=====================================
    optimizer_map = {}
    scheduler_map = {}
    
    # 根据模态数量和类型创建对应的优化器
    if args.optimizer == 'Adam':
        OptimClass = torch.optim.Adam
    elif args.optimizer == 'Adamw':
        OptimClass = torch.optim.AdamW
    elif args.optimizer == 'SGD':
        OptimClass = torch.optim.SGD
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # 为每个编码器创建优化器（使用模态名称作为key）
    if hasattr(model, 'encoder_1') and model.encoder_1 is not None:
        lr_key = 'learning_rate_' + modal_names[0].lower()
        lr = getattr(args, lr_key, args.learning_rate_fusion)
        encoder_key = f'encoder_{modal_names[0].lower()}'
        optimizer_map[encoder_key] = OptimClass(
            model.encoder_1.parameters(), lr=lr, weight_decay=args.weight_decay
        )
        scheduler_map[encoder_key] = torch.optim.lr_scheduler.StepLR(
            optimizer_map[encoder_key], args.lr_decay_step, args.lr_decay_ratio
        )
    
    if hasattr(model, 'encoder_2') and model.encoder_2 is not None:
        lr_key = 'learning_rate_' + modal_names[1].lower()
        lr = getattr(args, lr_key, args.learning_rate_fusion)
        encoder_key = f'encoder_{modal_names[1].lower()}'
        optimizer_map[encoder_key] = OptimClass(
            model.encoder_2.parameters(), lr=lr, weight_decay=args.weight_decay
        )
        scheduler_map[encoder_key] = torch.optim.lr_scheduler.StepLR(
            optimizer_map[encoder_key], args.lr_decay_step, args.lr_decay_ratio
        )
    
    if len(modal_names) == 3 and hasattr(model, 'encoder_3') and model.encoder_3 is not None:
        lr_key = 'learning_rate_' + modal_names[2].lower()
        lr = getattr(args, lr_key, args.learning_rate_fusion)
        encoder_key = f'encoder_{modal_names[2].lower()}'
        optimizer_map[encoder_key] = OptimClass(
            model.encoder_3.parameters(), lr=lr, weight_decay=args.weight_decay
        )
        scheduler_map[encoder_key] = torch.optim.lr_scheduler.StepLR(
            optimizer_map[encoder_key], args.lr_decay_step, args.lr_decay_ratio
        )
    
    # 融合层优化器
    if hasattr(model, 'fusion_model') and model.fusion_model is not None:
        optimizer_map['fusion'] = OptimClass(
            model.fusion_model.parameters(), 
            lr=args.learning_rate_fusion, 
            weight_decay=args.weight_decay
        )
        scheduler_map['fusion'] = torch.optim.lr_scheduler.StepLR(
            optimizer_map['fusion'], args.lr_decay_step, args.lr_decay_ratio
        )
    
    # ==================日志设置=====================================
    save_path = os.path.join(args.ckpt_path, f"AMST_Joint_{args.dataset}_{args.model_save_name}")
    os.makedirs(save_path, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(save_path, 'config.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    # 创建CSV日志文件
    log_path = os.path.join(save_path, f'{args.dataset}_amst_joint.csv')
    with open(log_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train_Acc', 'Train_F1', 'Train_Loss', 'Train_Time',
                        'Val_Acc', 'Val_F1', 'Val_Loss'])
    
    # Tensorboard
    writer = None
    if args.use_tensorboard:
        if args.tensorboard_path is None:
            args.tensorboard_path = os.path.join(save_path, 'tensorboard')
        os.makedirs(args.tensorboard_path, exist_ok=True)
        writer = SummaryWriter(args.tensorboard_path)
    
    # 训练历史
    history = {
        'train_acc': [], 'train_f1': [], 'train_loss': [],
        'val_acc': [], 'val_f1': [], 'val_loss': [],
        'epoch_times': [], 'skip_history': {}
    }
    
    best_val_acc = 0.0
    save_model_path = None
    
    print(f"\n{'='*60}")
    print(f"Starting AMST Joint Training")
    print(f"Dataset: {args.dataset}")
    print(f"Modalities: {modal_names}")
    print(f"Skip Factors: {skip_factors}")
    print(f"{'='*60}\n")
    
    # ==================训练和验证=====================================
    if args.train:
        for epoch in range(args.current_epoch, args.epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*50}")
            
            # 训练
            train_results = train_epoch_amst_joint(
                args, epoch, model, device, train_loader, 
                optimizer_map, scheduler_map, skip_factors
            )
            
            # 验证
            val_results = valid_amst_joint(args, epoch, model, device, val_loader)
            
            # 记录历史
            history['train_acc'].append(train_results['acc_fusion'])
            history['train_f1'].append(train_results['f1_fusion'])
            history['train_loss'].append(train_results['loss_fusion'])
            history['val_acc'].append(val_results['acc_fusion'])
            history['val_f1'].append(val_results['f1_fusion'])
            history['val_loss'].append(val_results['loss_fusion'])
            history['epoch_times'].append(train_results.get('epoch_time', 0))
            
            # 记录跳过历史
            for modal_name in modal_names:
                if should_skip_modality(modal_name, epoch, skip_factors):
                    if modal_name not in history['skip_history']:
                        history['skip_history'][modal_name] = []
                    history['skip_history'][modal_name].append(epoch)
            
            # Tensorboard记录
            if writer is not None:
                writer.add_scalar('Train/Accuracy', train_results['acc_fusion'], epoch)
                writer.add_scalar('Train/F1', train_results['f1_fusion'], epoch)
                writer.add_scalar('Train/Loss', train_results['loss_fusion'], epoch)
                writer.add_scalar('Val/Accuracy', val_results['acc_fusion'], epoch)
                writer.add_scalar('Val/F1', val_results['f1_fusion'], epoch)
                writer.add_scalar('Val/Loss', val_results['loss_fusion'], epoch)
                
                for modal_name in modal_names:
                    if modal_name in train_results['modal_metrics']:
                        writer.add_scalar(f'Train/{modal_name}_Acc', 
                                        train_results['modal_metrics'][modal_name]['acc'], epoch)
                        writer.add_scalar(f'Val/{modal_name}_Acc',
                                        val_results['modal_metrics'][modal_name]['acc'], epoch)
            
            # 记录CSV日志
            with open(log_path, 'a', newline='') as csvfile:
                writer_csv = csv.writer(csvfile)
                writer_csv.writerow([epoch, train_results['acc_fusion'], train_results['f1_fusion'],
                               train_results['loss_fusion'], train_results.get('epoch_time', 0),
                               val_results['acc_fusion'], val_results['f1_fusion'], val_results['loss_fusion']])
            
            # 打印结果
            print(f"\nEpoch {epoch}/{args.epochs}:")
            print(f"  Train - Acc: {train_results['acc_fusion']:.4f}, F1: {train_results['f1_fusion']:.4f}, "
                  f"Loss: {train_results['loss_fusion']:.4f}, Time: {train_results.get('epoch_time', 0):.2f}s")
            print(f"  Val   - Acc: {val_results['acc_fusion']:.4f}, F1: {val_results['f1_fusion']:.4f}, "
                  f"Loss: {val_results['loss_fusion']:.4f}")
            
            for modal_name in modal_names:
                train_m = train_results['modal_metrics'].get(modal_name, {})
                val_m = val_results['modal_metrics'].get(modal_name, {})
                print(f"  {modal_name} - Train Acc: {train_m.get('acc', 0):.4f}, Val Acc: {val_m.get('acc', 0):.4f}")
            
            # 保存最佳模型
            if val_results['acc_fusion'] > best_val_acc:
                best_val_acc = val_results['acc_fusion']
                model_name = f"{args.model_save_name}_amst_joint_best_acc{best_val_acc:.4f}_epoch{epoch}.pth"
                save_model_path = os.path.join(save_path, model_name)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_map': {k: v.state_dict() for k, v in optimizer_map.items()},
                    'best_val_acc': best_val_acc,
                    'args': args,
                }, save_model_path)
                print(f"  >>> Saved best model with Val Acc: {best_val_acc:.4f}")
            
            # 定期保存检查点
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_map': {k: v.state_dict() for k, v in optimizer_map.items()},
                    'history': history,
                    'args': args,
                }, os.path.join(save_path, f'checkpoint_epoch_{epoch}.pth'))
        
        # 保存训练历史
        with open(os.path.join(save_path, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        
        # 打印跳过统计
        print(f"\n{'='*60}")
        print("SKIP STATISTICS:")
        for modal_name, skip_epochs in history['skip_history'].items():
            print(f"  {modal_name}: skipped {len(skip_epochs)} epochs")
            if len(skip_epochs) <= 20:
                print(f"    Epochs: {skip_epochs}")
        print(f"{'='*60}\n")
        
        # 打印训练时间统计
        print(f"{'='*60}")
        print(f"Training Time Statistics")
        print(f"{'='*60}")
        if history['epoch_times']:
            total_time = sum(history['epoch_times'])
            avg_time = total_time / len(history['epoch_times'])
            print(f"Total Training Time: {total_time:.2f}s ({total_time/60:.2f}min / {total_time/3600:.2f}h)")
            print(f"Average Epoch Time: {avg_time:.2f}s ({avg_time/60:.2f}min)")
        print(f"{'='*60}\n")
        
        # 写入结果日志
        results_dir = "Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        log_file = os.path.join(results_dir, f"results-AMST-Joint-{args.dataset}.log")
        with open(log_file, "a") as f:
            f.write(f"==================== {datetime.datetime.now()} ===================\n\n")
            f.write(f"========================={args.model_save_name}==================================\n")
            f.write(f"Best Val Acc: {best_val_acc:.4f}\n")
            f.write(f"Skip Factors: {skip_factors}\n")
            if history['epoch_times']:
                total_time = sum(history['epoch_times'])
                avg_time = total_time / len(history['epoch_times'])
                f.write(f"Total Training Time: {total_time:.2f}s ({total_time/60:.2f}min / {total_time/3600:.2f}h)\n")
                f.write(f"Average Epoch Time: {avg_time:.2f}s ({avg_time/60:.2f}min)\n")
            f.write(f"Best model saved to: {save_model_path}\n")
            f.write(f"Args: {args}\n\n")
        
        print(f"Best Val Acc: {best_val_acc:.4f}")
        if save_model_path:
            print(f"Model saved to: {save_model_path}")
        print(f"Results logged to: {log_file}")
    
    if writer is not None:
        writer.close()
    
    print("Training completed!")


if __name__ == '__main__':
    main()
