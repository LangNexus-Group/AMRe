import argparse
import os
import csv
import ast
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.dataloader import AV_CD_Dataset
from model.basic_model import VA_Classifier, TVA_Classifier
from dataset.Mydataset import TVADataset, M3AEDataset
from dataset.Mydataset import CramedDataset, AVEDataset, KSDataset
from dataset.ResampleDataset import (
    CramedDataset_modality_level,
    AVEDataset_modality_level,
    KSDataset_modality_level,
    M3AEDataset_modality_level,
    TVADataset_modality_level
)

from utils.metrics import calculate_metrics
from utils.utils import setup_seed, weight_init, print_model_params


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str, help='KineticSound, CREMAD, AVE, Food101, MVSA, IEMOCAP3')
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
    parser.add_argument('--Use_initWeight', default=False, type=bool, help='Use weight init model')
    
    # 模型设置
    parser.add_argument('--fusion_method', default='concat', type=str,
                       choices=['sum', 'concat', 'Gate', 'Film', 'share', 'CAfusion'],
                       help='Fusion method to combine modalities')
    parser.add_argument('--model_name', default='["Visual","Audio"]', type=str,
                       choices=['["Visual","Audio"]', '["Image","Text"]', '["Text","Visual","Audio"]'])
    parser.add_argument('--modality', default='full', type=str,
                       choices=['full', 'audio', 'visual'],
                       help='modality to use')
    parser.add_argument('--unified_dim', default=512, type=int, help='Unified feature dimension after encoders')
    parser.add_argument('--m1_token_len', default=1, type=int, help='Modality 1 token length')
    parser.add_argument('--m2_token_len', default=1, type=int, help='Modality 2 token length')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes')
    parser.add_argument('--m1_gate', default=False, type=bool, help='For Gate Model, whether to use modality 1 gate')
    parser.add_argument('--x_film', default=False, type=bool, help='For Film Model, whether to use modality 1 film')
    
    # AME相关参数
    parser.add_argument('--MaskType', default='None', type=str, choices=['None', 'AME'], help='Type of masking strategy')
    parser.add_argument('--alpha', default=1.0, type=float, help='Alpha parameter for AME module')
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
    
    # OGM-GE 模型相关参数
    parser.add_argument('--modulation', default='OGM_GE', type=str, choices=['OGM', 'OGM_GE'], help='Modulation strategy to use')
    parser.add_argument('--ogm_alpha', default=0.8, type=float, help='Alpha parameter for OGM modulation')
    parser.add_argument('--modulation_starts', default=0, type=int, help='Epoch to start applying OGM modulation')
    parser.add_argument('--modulation_ends', default=50, type=int, help='Epoch to stop applying OGM modulation')
    parser.add_argument('--Use_OGM', default=False, type=bool, help='Whether to use OGM modulation')
    
    # Resample相关参数
    parser.add_argument('--use_resample', default=False, type=bool, help='Whether to use Resample training')
    parser.add_argument('--resample_warmup', default=5, type=int, help='Number of warmup epochs before Resample')
    parser.add_argument('--part_ratio', default=0.2, type=float, help='Percentage of subset to estimate modality preference')
    parser.add_argument('--resample_alpha', default=1.0, type=float, help='Alpha parameter for Resample')
    parser.add_argument('--resample_func', default='linear', type=str, choices=['linear', 'tanh', 'square'], 
                       help='Difference calculation function')
    
    args = parser.parse_args()
    return args


def calculate_contribution(args, model, device, dataloader, epoch):
    """
    计算每个样本的模态贡献度
    支持两模态和三模态
    """
    modal_names = ast.literal_eval(args.model_name)
    num_modalities = len(modal_names)
    n_classes = args.num_classes
    
    contribution = {}
    softmax = nn.Softmax(dim=1)
    
    # 累计贡献度
    if num_modalities == 2:
        cona, conv = 0.0, 0.0
    elif num_modalities == 3:
        cona, conv, cont = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        model.eval()
        
        for step, data_packet in tqdm(enumerate(dataloader), desc="Calculating contribution"):
            if modal_names == ["Visual", "Audio"]:
                # spec, image, label, sid = data_packet[0], data_packet[1], data_packet[2], data_packet[4]
                spec, image, label = data_packet[0], data_packet[1], data_packet[2]
                
                spec, image, label = spec.to(device), image.to(device), label.to(device)
                if args.dataset == 'CREMAD':
                    data_mini_packet = (image.float(), spec.float())
                else:
                    data_mini_packet = (image.float(), spec.unsqueeze(1).float())
            elif modal_names == ["Image", "Text"]:
                if len(data_packet) >= 6:
                    tokenizer, padding_mask, image, label, index, *_ = data_packet
                else:
                    tokenizer, padding_mask, image, label, index = data_packet[:5]
                tokenizer = tokenizer.to(device)
                padding_mask = padding_mask.to(device)
                image = image.to(device)
                label = label.to(device)
                data_mini_packet = [tokenizer, padding_mask, image]
            elif modal_names == ["Text", "Visual", "Audio"]:
                if len(data_packet) >= 7:
                    tokenizer, padding_mask, image, audio, label, index, *_ = data_packet
                else:
                    tokenizer, padding_mask, image, audio, label, index = data_packet[:6]
                tokenizer = tokenizer.to(device)
                padding_mask = padding_mask.to(device)
                image = image.to(device)
                audio = audio.to(device)
                label = label.to(device)
                # data_mini_packet = [tokenizer, padding_mask, image, audio]
                data_mini_packet = [tokenizer, padding_mask, image.float(), audio.unsqueeze(1).float()]
                
            
            # 前向传播获取输出
            outputs = model(data_mini_packet, epoch=epoch, labels=label)
            fusion_logits = outputs[0]
            modal_logits = list(outputs[1:1 + num_modalities])
            
            # 计算softmax概率
            prediction = softmax(fusion_logits)
            modal_preds = [softmax(logits) for logits in modal_logits]
            
            # 计算每个样本的贡献度
            for i, lbl in enumerate(label):
                # 获取预测索引
                all_pred = torch.argmax(prediction[i]).item()
                modal_pred_indices = [torch.argmax(pred[i]).item() for pred in modal_preds]
                
                # 计算value
                value_all = 2.0 if all_pred == lbl.item() else 0.0
                modal_values = [1.0 if idx == lbl.item() else 0.0 for idx in modal_pred_indices]
                
                # 计算贡献度
                if num_modalities == 2:
                    # 两模态：audio和visual
                    contrib_m1 = (modal_values[0] + value_all - modal_values[1]) / 2.0
                    contrib_m2 = (modal_values[1] + value_all - modal_values[0]) / 2.0
                    cona += contrib_m1
                    conv += contrib_m2
                    # 使用step索引而不是index（避免index不存在的问题）
                    sample_idx = step * args.batch_size + i
                    contribution[sample_idx] = (contrib_m1, contrib_m2)
                    
                elif num_modalities == 3:
                    # 三模态：text, visual, audio
                    contrib_t = (modal_values[0] + value_all - (modal_values[1] + modal_values[2]) / 2.0) / 2.0
                    contrib_v = (modal_values[1] + value_all - (modal_values[0] + modal_values[2]) / 2.0) / 2.0
                    contrib_a = (modal_values[2] + value_all - (modal_values[0] + modal_values[1]) / 2.0) / 2.0
                    cont += contrib_t
                    conv += contrib_v
                    cona += contrib_a
                    sample_idx = step * args.batch_size + i
                    contribution[sample_idx] = (contrib_t, contrib_v, contrib_a)
    
    # 计算平均贡献度
    total_samples = len(dataloader.dataset)
    if num_modalities == 2:
        cona /= total_samples
        conv /= total_samples
        print(f"\nEpoch {epoch} - Average Contribution: Modal1={cona:.4f}, Modal2={conv:.4f}")
        return contribution, (cona, conv)
    elif num_modalities == 3:
        cont /= total_samples
        conv /= total_samples
        cona /= total_samples
        print(f"\nEpoch {epoch} - Average Contribution: Text={cont:.4f}, Visual={conv:.4f}, Audio={cona:.4f}")
        return contribution, (cont, conv, cona)


def execute_resample(args, model, device, dataloader, epoch, contribution_history):
    """
    执行重采样，创建模态级重采样的数据集
    """
    resample_start_time = time.time()
    modal_names = ast.literal_eval(args.model_name)
    num_modalities = len(modal_names)
    
    # 计算当前epoch的贡献度
    contribution, avg_contrib = calculate_contribution(args, model, device, dataloader, epoch)
    contribution_history.append(avg_contrib)
    
    # 如果在warmup阶段，不进行重采样
    if epoch < args.resample_warmup:
        resample_time = time.time() - resample_start_time
        print(f"Epoch {epoch}: Warmup phase, no resampling.")
        print(f"  Contribution calculation time: {resample_time:.2f}s ({resample_time/60:.2f}min)")
        return None, contribution_history, resample_time
    
    # 计算部分样本的平均贡献度（使用part_ratio）
    num_samples = len(dataloader.dataset)
    num_part = int(num_samples * args.part_ratio)
    choice = np.random.choice(num_samples, num_part, replace=False)
    
    if num_modalities == 2:
        part_contrib1, part_contrib2 = 0.0, 0.0
        for i in choice:
            if i in contribution:
                c1, c2 = contribution[i]
                part_contrib1 += c1
                part_contrib2 += c2
        part_contrib1 /= num_part
        part_contrib2 /= num_part
        
        print(f"\nEpoch {epoch} - Partial Contribution (ratio={args.part_ratio}):")
        print(f"  Modal1={part_contrib1:.4f}, Modal2={part_contrib2:.4f}")
        
        # 创建重采样数据集
        if args.dataset == 'CREMAD':
            resampled_dataset = CramedDataset_modality_level(
                args=args,
                mode='train',
                contribution_a=part_contrib1,
                contribution_v=part_contrib2,
                alpha=args.resample_alpha,
                func=args.resample_func
            )
        elif args.dataset == 'AVE':
            resampled_dataset = AVEDataset_modality_level(
                args=args,
                mode='train',
                contribution_a=part_contrib1,
                contribution_v=part_contrib2,
                alpha=args.resample_alpha,
                func=args.resample_func
            )
        elif args.dataset == 'KineticSound':
            resampled_dataset = KSDataset_modality_level(
                args=args,
                mode='train',
                contribution_a=part_contrib1,
                contribution_v=part_contrib2,
                alpha=args.resample_alpha,
                func=args.resample_func
            )
        elif args.dataset == 'MVSA' or args.dataset == 'Food101':
            resampled_dataset = M3AEDataset_modality_level(
                args=args,
                mode='train',
                contribution_t=part_contrib1,
                contribution_v=part_contrib2,
                alpha=args.resample_alpha,
                func=args.resample_func
            )

        else:
            raise NotImplementedError(f"Resample not implemented for dataset: {args.dataset}")
            
    elif num_modalities == 3:
        part_contrib1, part_contrib2, part_contrib3 = 0.0, 0.0, 0.0
        for i in choice:
            if i in contribution:
                c1, c2, c3 = contribution[i]
                part_contrib1 += c1
                part_contrib2 += c2
                part_contrib3 += c3
        part_contrib1 /= num_part
        part_contrib2 /= num_part
        part_contrib3 /= num_part
        
        print(f"\nEpoch {epoch} - Partial Contribution (ratio={args.part_ratio}):")
        print(f"  Text={part_contrib1:.4f}, Visual={part_contrib2:.4f}, Audio={part_contrib3:.4f}")
        
        # 创建重采样数据集

        if args.dataset == 'IEMOCAP3':
            resampled_dataset = TVADataset_modality_level(
                args=args,
                mode='train',
                pick_num=3,
                contribution_a=part_contrib3,
                contribution_v=part_contrib2,
                contribution_t=part_contrib1,
                alpha=args.resample_alpha,
                func=args.resample_func
            )
        else:
            raise NotImplementedError(f"Resample not implemented for dataset: {args.dataset}")
    
    # 创建新的DataLoader
    resampled_dataloader = DataLoader(
        resampled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    resample_time = time.time() - resample_start_time
    print(f"Resampled dataset size: {len(resampled_dataset)}")
    print(f"  Resample time: {resample_time:.2f}s ({resample_time/60:.2f}min)")
    return resampled_dataloader, contribution_history, resample_time


def train_epoch(args, epoch, model, device, dataloader, optimizer_m1, optimizer_m2, optimizer_fusion, 
               optimizer_m3=None, scheduler_map=None):
    """标准训练函数"""
    epoch_start_time = time.time()
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.train()
    
    modal_names = ast.literal_eval(args.model_name)
    num_modalities = len(modal_names)
    
    all_acc_fusion, all_f1_fusion = [], []
    modal_acc_lists = [[] for _ in range(num_modalities)]
    modal_f1_lists = [[] for _ in range(num_modalities)]
    modal_loss_sums = [0.0 for _ in range(num_modalities)]
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs} [Training]")
    for step, data_packet in enumerate(pbar):
        # 清零梯度
        if optimizer_m1 is not None:
            optimizer_m1.zero_grad()
        if optimizer_m2 is not None:
            optimizer_m2.zero_grad()
        if optimizer_m3 is not None:
            optimizer_m3.zero_grad()
        if optimizer_fusion is not None:
            optimizer_fusion.zero_grad()
        
        # 处理数据
        drop = None
        if modal_names == ["Visual", "Audio"]:
            
            spec, image, label = data_packet[0], data_packet[1], data_packet[2]
            spec, image, label = spec.to(device), image.to(device), label.to(device)
            if args.dataset == 'CREMAD':
                data_mini_packet = (image.float(), spec.float())
                if len(data_packet) >= 6:
                    drop = data_packet[-1]
                    if isinstance(drop, torch.Tensor):
                        drop = drop.to(device)
            else:
                data_mini_packet = (image.float(), spec.unsqueeze(1).float())
                if len(data_packet) >= 5:
                    drop = data_packet[-1]
                    if isinstance(drop, torch.Tensor):
                        drop = drop.to(device)
            
        elif modal_names == ["Image", "Text"]:
            if len(data_packet) >= 6:
                tokenizer, padding_mask, image, label, index, drop = data_packet[:6]
                if isinstance(drop, torch.Tensor):
                    drop = drop.to(device)
            else:
                tokenizer, padding_mask, image, label, index = data_packet[:5]
            tokenizer = tokenizer.to(device)
            padding_mask = padding_mask.to(device)
            image = image.to(device)
            label = label.to(device)
            data_mini_packet = [tokenizer, padding_mask, image]
            
        elif modal_names == ["Text", "Visual", "Audio"]:
            if len(data_packet) >= 7:
                tokenizer, padding_mask, image, audio, label, index, drop = data_packet[:7]
                if isinstance(drop, torch.Tensor):
                    drop = drop.to(device)
            else:
                tokenizer, padding_mask, image, audio, label, index = data_packet[:6]
            tokenizer = tokenizer.to(device)
            padding_mask = padding_mask.to(device)
            image = image.to(device)
            audio = audio.to(device)
            label = label.to(device)
            data_mini_packet = [tokenizer, padding_mask, image.float(), audio.unsqueeze(1).float()]
        
        # 前向传播
        if drop is not None:
            # if step == 0:
            #     print(f"drop is {drop}")
            outputs = model(data_mini_packet, epoch=epoch, labels=label, epoch_index=step, drop=drop)
        else:
            outputs = model(data_mini_packet, epoch=epoch, labels=label, epoch_index=step)
        
        fusion_logits = outputs[0]
        modal_logits = list(outputs[1:1 + num_modalities])
        extra = list(outputs[1 + num_modalities:])
        
        # 处理mask
        masks = []
        if extra:
            masks = extra[:num_modalities] if len(extra) >= num_modalities else extra
        masks = [m.float() if isinstance(m, torch.Tensor) else None for m in masks]
        
        # 计算指标
        acc_fusion, f1_fusion = calculate_metrics(fusion_logits, label)
        all_acc_fusion.append(acc_fusion)
        all_f1_fusion.append(f1_fusion)
        
        # 计算损失
        modal_losses_each = [criterion(logit, label) for logit in modal_logits]
        weighted_modal_losses = []
        modal_loss_values = []
        
        for idx, (loss_each, mask) in enumerate(zip(modal_losses_each, masks + [None] * (num_modalities - len(masks)))):
            if mask is not None:
                weighted_loss = (loss_each * mask).sum() / (mask.sum() + 1e-8)
            else:
                weighted_loss = loss_each.mean()
            weighted_modal_losses.append(weighted_loss)
            modal_loss_values.append(weighted_loss.item())
            modal_loss_sums[idx] += weighted_loss.item()
        
        # 计算模态准确率
        for idx, logits in enumerate(modal_logits):
            acc_i, f1_i = calculate_metrics(logits, label)
            modal_acc_lists[idx].append(acc_i)
            modal_f1_lists[idx].append(f1_i)
        
        # 总损失
        loss_fusion = criterion(fusion_logits, label).mean()
        if args.modality == 'full':
            loss = sum(weighted_modal_losses) * args.alpha + loss_fusion
        else:
            loss = loss_fusion
        
        # 反向传播
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)
        
        if optimizer_m1 is not None:
            optimizer_m1.step()
        if optimizer_m2 is not None:
            optimizer_m2.step()
        if optimizer_m3 is not None:
            optimizer_m3.step()
        if optimizer_fusion is not None:
            optimizer_fusion.step()
        
        # 更新进度条
        postfix = {
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{acc_fusion:.4f}',
            'F1': f'{f1_fusion:.4f}',
        }
        for name, acc_i, loss_val in zip(modal_names, [l[-1] for l in modal_acc_lists], modal_loss_values):
            postfix[f'{name}_Acc'] = f'{acc_i:.4f}'
            postfix[f'{name}_Loss'] = f'{loss_val:.4f}'
        pbar.set_postfix(postfix)
    
    # 计算平均指标
    avg_acc_fusion = sum(all_acc_fusion) / len(all_acc_fusion) if all_acc_fusion else 0.0
    avg_f1_fusion = sum(all_f1_fusion) / len(all_f1_fusion) if all_f1_fusion else 0.0
    
    modal_avg_metrics = {}
    for name, acc_list, f1_list in zip(modal_names, modal_acc_lists, modal_f1_lists):
        avg_acc = sum(acc_list) / len(acc_list) if acc_list else 0.0
        avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
        modal_avg_metrics[name] = (avg_acc, avg_f1)
    
    epoch_time = time.time() - epoch_start_time
    print(f"\nTrain Epoch {epoch} Summary:")
    print(f"  Fusion -> Accuracy: {avg_acc_fusion:.4f}, F1-Score: {avg_f1_fusion:.4f}")
    for name in modal_names:
        acc, f1 = modal_avg_metrics.get(name, (0.0, 0.0))
        print(f"  {name} -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    print(f"  Epoch Time: {epoch_time:.2f}s ({epoch_time/60:.2f}min)")
    
    return (avg_acc_fusion, avg_f1_fusion), modal_avg_metrics, epoch_time


def valid(args, model, device, dataloader):
    """验证函数"""
    with torch.no_grad():
        model.eval()
        
        modal_names = ast.literal_eval(args.model_name)
        num_modalities = len(modal_names)
        
        all_labels = []
        fusion_preds = []
        modal_preds = [[] for _ in range(num_modalities)]
        
        pbar = tqdm(dataloader, desc="Validating")
        for step, data_packet in enumerate(pbar):
            if modal_names == ["Visual", "Audio"]:
                # spec, image, label, sid = data_packet[0], data_packet[1], data_packet[2], data_packet[4]
                spec, image, label = data_packet[0], data_packet[1], data_packet[2]
                
                spec, image, label = spec.to(device), image.to(device), label.to(device)
                if args.dataset == 'CREMAD':
                    data_mini_packet = (image.float(), spec.float())
                else:
                    data_mini_packet = (image.float(), spec.unsqueeze(1).float())
            elif modal_names == ["Image", "Text"]:
                tokenizer, padding_mask, image, label, index = data_packet[:5]
                tokenizer = tokenizer.to(device)
                padding_mask = padding_mask.to(device)
                image = image.to(device)
                label = label.to(device)
                data_mini_packet = [tokenizer, padding_mask, image]
            elif modal_names == ["Text", "Visual", "Audio"]:
                tokenizer, padding_mask, image, audio, label, index = data_packet[:6]
                tokenizer = tokenizer.to(device)
                padding_mask = padding_mask.to(device)
                image = image.to(device)
                audio = audio.to(device)
                label = label.to(device)
                # data_mini_packet = [tokenizer, padding_mask, image, audio]
                data_mini_packet = [tokenizer, padding_mask, image.float(), audio.unsqueeze(1).float()]
                
            
            outputs = model(data_mini_packet, epoch=0, labels=label)
            
            fusion_logits = outputs[0]
            modal_logits = list(outputs[1:1 + num_modalities])
            
            all_labels.append(label)
            fusion_preds.append(fusion_logits)
            for idx, logits in enumerate(modal_logits):
                modal_preds[idx].append(logits)
        
        # 合并所有batch的结果
        fusion_preds = torch.cat(fusion_preds, dim=0)
        modal_preds = [torch.cat(preds, dim=0) for preds in modal_preds]
        all_labels = torch.cat(all_labels, dim=0)
        
        # 计算指标
        acc_fusion, f1_fusion = calculate_metrics(fusion_preds, all_labels)
        modal_metrics = {}
        for name, preds in zip(modal_names, modal_preds):
            acc, f1 = calculate_metrics(preds, all_labels)
            modal_metrics[name] = (acc, f1)
        
        print(f"\nValidation Summary:")
        print(f"  Fusion -> Accuracy: {acc_fusion:.4f}, F1-Score: {f1_fusion:.4f}")
        for name in modal_names:
            acc, f1 = modal_metrics.get(name, (0.0, 0.0))
            print(f"  {name} -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    
    return (acc_fusion, f1_fusion), modal_metrics


def main():
    args = get_arguments()
    print(args)
    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 数据集加载
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
    
    # 模型初始化
    if args.model_name == '["Visual","Audio"]' or args.model_name == '["Image","Text"]':
        model = VA_Classifier(args)
    else:
        model = TVA_Classifier(args)
    
    print_model_params(model)
    model.to(device)
    if args.model_name == '["Visual","Audio"]' and args.Use_initWeight:
        model.apply(weight_init)
        print(f"Use Weight int")
    
    # 优化器设置
    optimizer_m1_params, optimizer_m2_params, optimizer_fusion_params = [], [], []
    optimizer_m3_params = []
    
    for name, param in model.named_parameters():
        if 'encoder_1' in name or 'm1' in name:
            optimizer_m1_params.append(param)
        elif 'encoder_2' in name or 'm2' in name:
            optimizer_m2_params.append(param)
        elif 'encoder_3' in name or 'm3' in name:
            optimizer_m3_params.append(param)
        else:
            optimizer_fusion_params.append(param)
    
    optimizer_m1, optimizer_m2, optimizer_m3 = None, None, None
    
    if args.optimizer == "Adamw":
        if args.fusion_method == 'sum':
            optimizer_fusion = torch.optim.AdamW(model.parameters(), lr=args.learning_rate_fusion,
                                                betas=(0.9, 0.999), weight_decay=args.weight_decay)
        else:
            modal_names = ast.literal_eval(args.model_name)
            if modal_names == ["Visual", "Audio"]:
                optimizer_m1 = torch.optim.AdamW(optimizer_m1_params, lr=args.learning_rate_visual,
                                                betas=(0.9, 0.999), weight_decay=args.weight_decay)
                optimizer_m2 = torch.optim.AdamW(optimizer_m2_params, lr=args.learning_rate_audio,
                                                betas=(0.9, 0.999), weight_decay=args.weight_decay)
            elif modal_names == ["Image", "Text"]:
                optimizer_m1 = torch.optim.AdamW(optimizer_m1_params, lr=args.learning_rate_image,
                                                betas=(0.9, 0.999), weight_decay=args.weight_decay)
                optimizer_m2 = torch.optim.AdamW(optimizer_m2_params, lr=args.learning_rate_text,
                                                betas=(0.9, 0.999), weight_decay=args.weight_decay)
            elif modal_names == ["Text", "Visual", "Audio"]:
                optimizer_m1 = torch.optim.AdamW(optimizer_m1_params, lr=args.learning_rate_text,
                                                betas=(0.9, 0.999), weight_decay=args.weight_decay)
                optimizer_m2 = torch.optim.AdamW(optimizer_m2_params, lr=args.learning_rate_visual,
                                                betas=(0.9, 0.999), weight_decay=args.weight_decay)
                optimizer_m3 = torch.optim.AdamW(optimizer_m3_params, lr=args.learning_rate_audio,
                                                betas=(0.9, 0.999), weight_decay=args.weight_decay)
            
            if optimizer_fusion_params:
                optimizer_fusion = torch.optim.AdamW(optimizer_fusion_params, lr=args.learning_rate_fusion,
                                                    betas=(0.9, 0.999), weight_decay=args.weight_decay)
            else:
                optimizer_fusion = None
    
    # 学习率调度器
    scheduler_map = None
    if args.fusion_method == 'sum':
        scheduler_fusion = torch.optim.lr_scheduler.StepLR(optimizer_fusion, args.lr_decay_step, args.lr_decay_ratio)
        scheduler_map = [scheduler_fusion]
    elif args.model_name == '["Text","Visual","Audio"]':
        scheduler_m1 = torch.optim.lr_scheduler.StepLR(optimizer_m1, args.lr_decay_step, args.lr_decay_ratio)
        scheduler_m2 = torch.optim.lr_scheduler.StepLR(optimizer_m2, args.lr_decay_step, args.lr_decay_ratio)
        scheduler_m3 = torch.optim.lr_scheduler.StepLR(optimizer_m3, args.lr_decay_step, args.lr_decay_ratio)
        if optimizer_fusion is not None:
            scheduler_fusion = torch.optim.lr_scheduler.StepLR(optimizer_fusion, args.lr_decay_step, args.lr_decay_ratio)
            scheduler_map = [scheduler_m1, scheduler_m2, scheduler_m3, scheduler_fusion]
        else:
            scheduler_map = [scheduler_m1, scheduler_m2, scheduler_m3]
    else:
        scheduler_m1 = torch.optim.lr_scheduler.StepLR(optimizer_m1, args.lr_decay_step, args.lr_decay_ratio)
        scheduler_m2 = torch.optim.lr_scheduler.StepLR(optimizer_m2, args.lr_decay_step, args.lr_decay_ratio)
        if optimizer_fusion is not None:
            scheduler_fusion = torch.optim.lr_scheduler.StepLR(optimizer_fusion, args.lr_decay_step, args.lr_decay_ratio)
            scheduler_map = [scheduler_m1, scheduler_m2, scheduler_fusion]
        else:
            scheduler_map = [scheduler_m1, scheduler_m2]
    
    # 创建保存路径
    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    
    log_path = os.path.join(args.ckpt_path, args.dataset + '_resample.csv')
    modal_names = ast.literal_eval(args.model_name)
    modal_header = []
    for name in modal_names:
        modal_header.extend([f'{name}_Acc', f'{name}_F1'])
    
    with open(log_path, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Train_Acc', 'Train_F1'] + modal_header + 
                       ['Val_Acc', 'Val_F1'] + modal_header)
    
    # 训练循环
    if args.train:
        best_acc = 0.0
        contribution_history = []
        epoch_times = []  # 记录每个epoch的训练时间
        resample_times = []  # 记录每个epoch的重采样时间
        
        for epoch in range(args.current_epoch, args.epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{args.epochs}")
            print(f"{'='*50}")
            
            epoch_total_start = time.time()  # 记录整个epoch的开始时间
            
            # Warmup阶段或非Resample模式：正常训练
            # if not args.use_resample or epoch < args.resample_warmup or epoch%2==1:
            if not args.use_resample or epoch < args.resample_warmup:
                print(f"Training warmup at epoch {epoch}")
                train_metrics, train_modal_metrics, train_time = train_epoch(
                    args, epoch, model, device, train_dataloader,
                    optimizer_m1, optimizer_m2, optimizer_fusion, optimizer_m3, scheduler_map
                )
                resample_time = 0.0  # warmup阶段没有resample
                resample_times.append(resample_time)
            else:
                # Resample阶段：先计算贡献度并重采样，然后训练
                print(f"Resample training at epoch {epoch}")
                resampled_dataloader, contribution_history, resample_time = execute_resample(
                    args, model, device, train_dataloader, epoch, contribution_history
                )
                resample_times.append(resample_time)
                
                if resampled_dataloader is not None:
                    train_metrics, train_modal_metrics, train_time = train_epoch(
                        args, epoch, model, device, resampled_dataloader,
                        optimizer_m1, optimizer_m2, optimizer_fusion, optimizer_m3, scheduler_map
                    )
                else:
                    train_metrics, train_modal_metrics, train_time = train_epoch(
                        args, epoch, model, device, train_dataloader,
                        optimizer_m1, optimizer_m2, optimizer_fusion, optimizer_m3, scheduler_map
                    )
            
            # 计算完整的epoch时间（resample + training）
            epoch_total_time = time.time() - epoch_total_start
            epoch_times.append(epoch_total_time)
            
            # 输出epoch总时间统计
            print(f"\nEpoch {epoch} Total Time Summary:")
            if resample_times[-1] > 0:
                print(f"  Resample Time: {resample_times[-1]:.2f}s ({resample_times[-1]/60:.2f}min)")
                print(f"  Training Time: {train_time:.2f}s ({train_time/60:.2f}min)")
            print(f"  Total Epoch Time: {epoch_total_time:.2f}s ({epoch_total_time/60:.2f}min)")
            
            # 更新学习率
            if scheduler_map is not None:
                for scheduler in scheduler_map:
                    scheduler.step()
            
            # 验证
            val_metrics, val_modal_metrics = valid(args, model, device, test_dataloader)
            
            # 记录日志
            train_acc, train_f1 = train_metrics
            val_acc, val_f1 = val_metrics
            
            train_modal_values = []
            val_modal_values = []
            for name in modal_names:
                t_acc, t_f1 = train_modal_metrics.get(name, (0.0, 0.0))
                v_acc, v_f1 = val_modal_metrics.get(name, (0.0, 0.0))
                train_modal_values.extend([t_acc, t_f1])
                val_modal_values.extend([v_acc, v_f1])
            
            with open(log_path, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([epoch, train_acc, train_f1] + train_modal_values +
                              [val_acc, val_f1] + val_modal_values)
            
            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                model_name = f"{args.model_save_name}_resample_best_acc{val_acc:.4f}_epoch{epoch}.pth"
                save_path = os.path.join(args.ckpt_path, model_name)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_m1_state_dict': optimizer_m1.state_dict() if optimizer_m1 else None,
                    'optimizer_m2_state_dict': optimizer_m2.state_dict() if optimizer_m2 else None,
                    'optimizer_m3_state_dict': optimizer_m3.state_dict() if optimizer_m3 else None,
                    'optimizer_fusion_state_dict': optimizer_fusion.state_dict() if optimizer_fusion else None,
                    'best_acc': best_acc,
                    'args': args
                }, save_path)
                

                
                print(f"\n✓ Best model saved: {save_path}")
                print(f"  Best Accuracy: {best_acc:.4f}")
        
        # 输出训练时间统计
        print(f"\n{'='*50}")
        print(f"Training Time Statistics")
        print(f"{'='*50}")
        if epoch_times:
            total_time = sum(epoch_times)
            total_resample_time = sum(resample_times)
            total_train_time = total_time - total_resample_time
            avg_time = total_time / len(epoch_times)
            avg_resample_time = total_resample_time / len(resample_times) if resample_times else 0
            min_time = min(epoch_times)
            max_time = max(epoch_times)
            
            print(f"Total Time (All Epochs): {total_time:.2f}s ({total_time/60:.2f}min / {total_time/3600:.2f}h)")
            print(f"  - Total Resample Time: {total_resample_time:.2f}s ({total_resample_time/60:.2f}min / {total_resample_time/3600:.2f}h)")
            print(f"  - Total Training Time: {total_train_time:.2f}s ({total_train_time/60:.2f}min / {total_train_time/3600:.2f}h)")
            print(f"\nAverage Time per Epoch: {avg_time:.2f}s ({avg_time/60:.2f}min)")
            if avg_resample_time > 0:
                print(f"  - Average Resample Time: {avg_resample_time:.2f}s ({avg_resample_time/60:.2f}min)")
            print(f"Min Epoch Time: {min_time:.2f}s ({min_time/60:.2f}min)")
            print(f"Max Epoch Time: {max_time:.2f}s ({max_time/60:.2f}min)")
            
            # print(f"\nEpoch-wise Time Breakdown:")
            # for i, (total_t, resample_t) in enumerate(zip(epoch_times, resample_times), start=args.current_epoch):
            #     train_t = total_t - resample_t
            #     if resample_t > 0:
            #         print(f"  Epoch {i}: {total_t:.2f}s ({total_t/60:.2f}min) [Resample: {resample_t:.2f}s, Train: {train_t:.2f}s]")
            #     else:
            #         print(f"  Epoch {i}: {total_t:.2f}s ({total_t/60:.2f}min) [Train only]")
        print(f"{'='*50}\n")
        
        # 写入结果日志
        results_dir = "Results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        log_file = os.path.join(results_dir, f"results-Resample-{args.dataset}-{args.fusion_method}.log")
        with open(log_file, "a") as f:
            f.write(
                f"==================== {datetime.datetime.now()} ===================\n \n"
            )
            f.write(f"========================={args.model_save_name}==================================\n")
            f.write(f"val_acc: {best_acc}\n")
            f.write(f"all metric: {val_metrics}\n")
            f.write(f"best model save as {save_path}\n")
            if epoch_times:
                total_time = sum(epoch_times)
                total_resample_time = sum(resample_times)
                total_train_time = total_time - total_resample_time
                avg_time = total_time / len(epoch_times)
                f.write(f"total_time: {total_time:.2f}s ({total_time/60:.2f}min / {total_time/3600:.2f}h)\n")
                f.write(f"total_resample_time: {total_resample_time:.2f}s ({total_resample_time/60:.2f}min / {total_resample_time/3600:.2f}h)\n")
                f.write(f"total_training_time: {total_train_time:.2f}s ({total_train_time/60:.2f}min / {total_train_time/3600:.2f}h)\n")
                f.write(f"avg_epoch_time: {avg_time:.2f}s ({avg_time/60:.2f}min)\n")
            f.write(f"args: {args}\n \n")
        print(f"val best metrics is {val_metrics} \n")
        print(f"best model save as {save_path} \n")
        print(f"args:{args} \n \n")


if __name__ == '__main__':
    main()
