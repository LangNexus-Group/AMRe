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
import pytorch_warmup as warmup

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.dataloader import AV_CD_Dataset
from model.basic_model import VA_Classifier,TVA_Classifier
# from model.AVClass import AVClassifier
from dataset.Mydataset import TVADataset, M3AEDataset
from dataset.Mydataset import CramedDataset,AVEDataset,KSDataset
from dataset.RemixDataset import CramedDataset_Remix, AVEDataset_Remix, M3AEDataset_Remix, TVADataset_Remix

from utils.metrics import calculate_metrics
from utils.utils import setup_seed,weight_init,print_model_params,print_current_lrs
from utils.utils import Alignment,getAlpha_Learnable_Fitted # For LFM

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,help=' KineticSound, CREMAD, AVE, MSVD')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
    parser.add_argument('--train', action='store_true', help='turn on train mode')
    parser.add_argument('--use_video_frames', default=3, type=int, help='use how many frames for train')
    
    # 学习率定义
    parser.add_argument('--learning_rate_visual', default=5e-5, type=float, help='Learning rate for visual encoder')
    parser.add_argument('--learning_rate_audio', default=5e-5, type=float, help='Learning rate for audio encoder')
    parser.add_argument('--learning_rate_text', default=1e-5, type=float, help='Learning rate for text encoder')
    parser.add_argument('--learning_rate_image', default=1e-3, type=float, help='Learning rate for text encoder')
    parser.add_argument('--learning_rate_fusion', default=5e-5, type=float, help='Learning rate for fusion model')
    parser.add_argument('--optimizer', default='Adamw', type=str,help='Optimizer to use (SGD, Adam, Adamw)')
    parser.add_argument('--lr_decay_step',default=30, type=int)
    parser.add_argument('--lr_decay_ratio',default=0.1, type=float)
    parser.add_argument('--weight_decay',default=1e-4, type=float)
    # 基础设置
    parser.add_argument('--random_seed', default=42, type=int)
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU ids')
    parser.add_argument('--current_epoch', type=int, default=1,help="Start train epoch number")
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--Use_initWeight', default=False, type=bool, help='Use weight init model')
    

    # 模型设置
    parser.add_argument('--fusion_method', default='concat', type=str,choices=['sum', 'concat', 'Gate', 'Film', 'share', 'CAfusion'], help='Fusion method to combine modalities')
    parser.add_argument('--model_name', default='["Visual","Audio"]', type=str, choices=['["Visual","Audio"]', '["Image","Text"]', '["Text","Visual","Audio"]'])
    parser.add_argument('--modality',default='full',type=str,choices=['full','audio','visual'],help='modality to use')
    parser.add_argument('--unified_dim', default=512, type=int, help='Unified feature dimension after encoders')
    parser.add_argument('--m1_token_len', default=1, type=int, help='Modality 1 (e.g., visual) token length')
    parser.add_argument('--m2_token_len', default=1, type=int, help='Modality 2 (e.g., audio or text) token length')
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
    parser.add_argument('--Use_MACE',default=True, type=bool, help="是否使用MACE loss")
    # tensorboard相关参数
    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')
    # 保存模型的相关参数
    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument("--model_save_name",required=True,type=str,help='model save log name')
    # OGM-GE 模型相关参数
    parser.add_argument('--modulation', default='OGM_GE', type=str, choices=['OGM', 'OGM_GE'], help='Modulation strategy to use')
    parser.add_argument('--ogm_alpha', default=0.8, type=float, help='Alpha parameter for OGM modulation')
    parser.add_argument('--modulation_starts', default=0, type=int, help='Epoch to start applying OGM modulation')
    parser.add_argument('--modulation_ends', default=50, type=int, help='Epoch to stop applying OGM modulation')
    parser.add_argument('--Use_OGM', default=False, type=bool, help='Whether to use OGM modulation')
    
    # LFM相关参数
    parser.add_argument('--LFM', default=False, type=bool, help='Whether to use LFM method')
    
    # Remix相关参数
    parser.add_argument('--use_remix', default=False, type=bool, help='Whether to use Remix training')
    parser.add_argument('--remix_warmup', default=10, type=int, help='Number of warmup epochs before Remix')
    parser.add_argument('--modality_names', default='audio,visual', type=str, help='Comma-separated modality names for Remix')
    parser.add_argument('--modality_gammas', default='1.0,1.0', type=str, help='Comma-separated gamma values for each modality in Remix')
    
    args = parser.parse_args()
    return args


def kl_divergence(predictions, num_classes):
    """计算预测分布与均匀分布之间的KL散度"""
    uniform_distribution = np.full(num_classes, 1.0 / num_classes)
    predictions = np.clip(predictions, 1e-9, 1.0)
    kl_div = np.sum(predictions * np.log2(predictions / uniform_distribution))
    return kl_div


def remix(args, model, device, dataloader, epoch):
    """
    Remix数据集，根据KL散度将样本分配到不同的模态特定子集
    支持任意数量的模态
    """
    remix_start_time = time.time()
    # 解析模态名称
    modality_names = args.modality_names.split(',')
    num_modalities = len(modality_names)
    
    # 初始化每个模态的数据集列表
    modality_datasets = {name.strip(): [] for name in modality_names}
    if args.MaskType == 'AME':
        modality_datasets['FULL'] = []
    softmax = nn.Softmax(dim=1)
    
    print(f"\n=== Epoch {epoch}: Running Remix ===")
    print(f"Modalities: {modality_datasets.keys()}")

    with torch.no_grad():
        model.eval()
        for step, data_packet in enumerate(tqdm(dataloader, desc="Remix sampling")):
            modal_names = ast.literal_eval(args.model_name)
            
            # 解析数据包
            if modal_names == ["Visual", "Audio"]:
                spec, image, label,sid = data_packet[0], data_packet[1], data_packet[2],data_packet[-1]
                # sid = data_packet[4] if len(data_packet) > 4 else [str(i) for i in range(len(label))]
                spec, image = spec.to(device), image.to(device)
                if args.dataset == 'CREMAD':
                    data_mini_packet = (image.float(), spec.float())
                else:
                    data_mini_packet = (image.float(), spec.unsqueeze(1).float())
            elif modal_names == ["Image", "Text"]:
                token, padding_mask, image, label, sid = data_packet
                token, padding_mask, image = token.to(device), padding_mask.to(device), image.to(device)
                data_mini_packet = (token, padding_mask, image)
            elif modal_names == ["Text", "Visual", "Audio"]:
                token, padding_mask, image, spec, label, sid = data_packet
                token, padding_mask = token.to(device), padding_mask.to(device)
                image, spec = image.to(device), spec.to(device)
                data_mini_packet = (token, padding_mask, image.float(), spec.unsqueeze(1).float())
            else:
                raise NotImplementedError(f"Unsupported modal combination: {modal_names}")
            label = label.to(device)
            # 获取模型输出
            if args.MaskType == 'None':
                outputs = model(data_mini_packet)
                fusion_logits = outputs[0]
                modality_outputs = list(outputs[1:1 + num_modalities])
                
                # 计算每个模态的softmax预测
                modality_preds = [softmax(out_m) for out_m in modality_outputs]
                # print(f"sid is {sid}")
                # print(f"modality_preds shape is {len(modality_preds)}")
                # 对每个样本计算KL散度并分配到模态子集
                for i in range(len(label)):
                    kl_divs = []
                    for pred in modality_preds:
                        pred_numpy = pred[i].cpu().data.numpy()
                        kl_div = kl_divergence(pred_numpy, args.num_classes)
                        kl_divs.append(kl_div)
                    # print(f"kl_divs is {len(kl_divs)}")
                    # 将样本分配给KL散度最小的模态
                    min_kl_idx = np.argmin(kl_divs)
                    # min_kl_idx = np.argmax(kl_divs)
                    selected_modality = modality_names[min_kl_idx].strip()  
                    # 获取样本标识和标签
                    sample_id = sid[i] if isinstance(sid, (list, tuple)) else str(sid[i].item())
                    sample_label = label[i].item() if torch.is_tensor(label) else label[i]
                    
                    modality_datasets[selected_modality].append([sample_id, sample_label])
                    
            else:
                # 设计满足任意条件
                with torch.no_grad():
                    # outputs = model.get_Mask(data_mini_packet, epoch=epoch, labels=label, epoch_index=step)
                    outputs = model.get_Mask(data_mini_packet, epoch=epoch, labels=label, epoch_index=-1)
    
                    for i in range(len(label)):
                        # 假设保留的是1
                        min_kl_idx=0
                        Mask = []
                        for j in range(num_modalities):
                            Mask.append(outputs[j][i])
                            # if outputs[j][i]:
                            #     min_kl_idx=j
    
                        if sum(Mask) == num_modalities:
                            selected_modality = 'FULL'
                        else:
                            min_kl_idx = Mask.index(1)
                            selected_modality = modality_names[min_kl_idx].strip()  
                        # 获取样本标识和标签
                        # if i==0 and step==0:
                        #     print(f"Mask is {Mask} selected_modality is {selected_modality}")
                        sample_id = sid[i] if isinstance(sid, (list, tuple)) else str(sid[i].item())
                        sample_label = label[i].item() if torch.is_tensor(label) else label[i]
                        
                        modality_datasets[selected_modality].append([sample_id, sample_label])
                # if step==0:
                #     print(f"m1_mask is {outputs[0].sum()} len m1 is {len(modality_datasets[modality_names[0]])}.")
                #     print(f"m2_mask is {outputs[1].sum()} len m1 is {len(modality_datasets[modality_names[1]])}.")
                    
                        
            
    # if args.MaskType == "AME":
    #     print(f"m1_feature mask is {model.fusion_model.m1_mask_num.item()}")
    #     print(f"m2_feature mask is {model.fusion_model.m2_mask_num.item()}")
    # 保存每个模态的数据集到CSV文件
    os.makedirs(f'./data/{args.dataset}', exist_ok=True)
    for modality_name in modality_datasets.keys():
        modality_name = modality_name.strip()
        file_path = f"./data/{args.dataset}/remix_{modality_name}.csv"
        with open(file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            for item in modality_datasets[modality_name]:
                writer.writerow([item[0], item[1]])
        print(f"Saved {len(modality_datasets[modality_name])} samples to {file_path}")

    # 创建每个模态的数据集对象
    remix_datasets = {}
    modal_names = ast.literal_eval(args.model_name)
    
    for modality_name in modality_datasets.keys():
        modality_name = modality_name.strip()
        try:
            if args.dataset == 'CREMAD':
                # print(f"USe CREMAD")
                remix_datasets[modality_name] = CramedDataset_Remix(args, modality=modality_name)
            elif args.dataset == 'AVE':
                remix_datasets[modality_name] = AVEDataset_Remix(args, modality=modality_name)
            elif args.dataset in ['Food101', 'MVSA']:
                remix_datasets[modality_name] = M3AEDataset_Remix(args, modality=modality_name)
            elif args.dataset == 'IEMOCAP3':
                remix_datasets[modality_name] = TVADataset_Remix(args, modality=modality_name, pick_num=3)
            else:
                # 默认处理：假设有通用的 Remix 数据集类
                print(f"Warning: No specific Remix dataset for {args.dataset}, using default")
                continue
        except Exception as e:
            print(f"Error creating Remix dataset for {modality_name}: {e}")
            continue

    remix_time = time.time() - remix_start_time
    print(f"  Remix total time: {remix_time:.2f}s ({remix_time/60:.2f}min)")
    return remix_datasets, remix_time


def special_train_epoch(args, epoch, model, device, modality_datasets, optimizer_m1, optimizer_m2, 
                       optimizer_fusion, optimizer_m3=None):
    """
    模态特定训练函数，支持任意数量的模态
    每个模态使用各自的数据子集进行训练
    """
    epoch_start_time = time.time()
    # 解析模态名称和对应的gamma值
    modality_names = [name.strip() for name in args.modality_names.split(',')]
    modality_gammas = [float(g) for g in args.modality_gammas.split(',')]
    
    if len(modality_gammas) != len(modality_names):
        print(f"Warning: gamma数量({len(modality_gammas)})与模态数量({len(modality_names)})不匹配，使用默认值1.0")
        modality_gammas = [1.0] * len(modality_names)
    if "FULL" in modality_datasets.keys():
        modality_gammas.append(int(args.alpha))
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.train()
    print(f"\nEpoch {epoch}: Modality-specific training...")
    _loss = 0
    total_batches = 0
    modal_names = ast.literal_eval(args.model_name)
    
    num_modalities = len(modal_names)
    # 遍历每个模态的数据集
    for modality_idx, modality_name in enumerate(modality_datasets.keys()):
        modality_name = modality_name.strip()
        if modality_name not in modality_datasets:
            print(f"Warning: {modality_name} not in modality_datasets, skipping")
            continue
            
        dataset = modality_datasets[modality_name]
        gamma = modality_gammas[modality_idx]
        
        if len(dataset) > 0:
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=4, 
                pin_memory=True
            )
            num_batches = len(dataloader)
            total_batches += num_batches
            
            print(f"Training on {modality_name} subset: {len(dataset)} samples, {num_batches} batches")
            
            pbar = tqdm(dataloader, desc=f"{modality_name} modality")
            for step, data_packet in enumerate(pbar):
                if optimizer_m1 is not None:
                    optimizer_m1.zero_grad()
                if optimizer_m2 is not None:
                    optimizer_m2.zero_grad()
                if optimizer_m3 is not None:
                    optimizer_m3.zero_grad()
                if optimizer_fusion is not None:
                    optimizer_fusion.zero_grad()

                # 解析数据包
                if modal_names == ["Visual", "Audio"]:
                    spec, image, label = data_packet[0], data_packet[1], data_packet[2]
                    spec, image, label = spec.to(device), image.to(device), label.to(device)
                    if args.dataset == 'CREMAD':
                        data_mini_packet = (image.float(), spec.float())
                    else:
                        data_mini_packet = (image.float(), spec.unsqueeze(1).float())
                elif modal_names == ["Image", "Text"]:
                    token, padding_mask, image, label, _, _ = data_packet
                    token, padding_mask = token.to(device), padding_mask.to(device)
                    image, label = image.to(device), label.to(device)
                    data_mini_packet = (token, padding_mask, image)
                elif modal_names == ["Text", "Visual", "Audio"]:
                    token, padding_mask, image, spec, label, _, _ = data_packet
                    token, padding_mask = token.to(device), padding_mask.to(device)
                    image, spec, label = image.to(device), spec.to(device), label.to(device)
                    data_mini_packet = (token, padding_mask, image.float(), spec.unsqueeze(1).float())
                else:
                    raise NotImplementedError(f"Unsupported modal combination: {modal_names}")
                
                # 前向传播（不使用模态masking，让模型自然处理）
                if modality_name == 'FULL':
                    modality_train_idx=None
                else:
                    modality_train_idx = modal_names.index(modality_name) + 1
                # if step == 0:
                #     print(f"modality_train_idx is {modality_train_idx}")
                outputs = model(data_mini_packet, epoch=0, labels=label, epoch_index=step,modality_idx=modality_train_idx)
                
                fusion_logits = outputs[0]
                modal_logits = list(outputs[1:1 + num_modalities])
                
                # 计算损失
                loss_fusion = criterion(fusion_logits, label).mean()
                modal_losses = [criterion(logit, label).mean() for logit in modal_logits]
                
                # 对当前训练的模态应用gamma权重
                if modality_name == 'FULL':
                    loss_single = sum(modal_losses) * args.alpha
                    loss = loss_fusion + loss_single
                else:
                    loss = loss_fusion + gamma * modal_losses[modality_idx]
                
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
                
                _loss += loss.item()
                
                # 更新tqdm进度条显示
                postfix = {
                    'Loss': f'{loss.item():.4f}',
                    'Fusion_Loss': f'{loss_fusion.item():.4f}',
                }
                for idx, name in enumerate(modal_names):
                    if idx < len(modal_losses):
                        postfix[f'{name}_Loss'] = f'{modal_losses[idx].item():.4f}'
                pbar.set_postfix(postfix)
        else:
            print(f"Empty {modality_name} Subset!")
    
    epoch_time = time.time() - epoch_start_time
    avg_loss = _loss / total_batches if total_batches > 0 else 0.0
    
    if total_batches == 0:
        print("Warning: No data in any modality subset!")
    
    print(f"\nEpoch {epoch} Remix Training Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Epoch Time: {epoch_time:.2f}s ({epoch_time/60:.2f}min)")
        
    return avg_loss, epoch_time



def train_epoch(args, epoch, model, device, dataloader, optimizer_m1, optimizer_m2, optimizer_fusion, optimizer_m3=None,scheduler_map=None):
    epoch_start_time = time.time()
    criterion = nn.CrossEntropyLoss(reduction='none')
    model.train()
    cls_k = None
    if args.LFM:
        cls_k = getAlpha_Learnable_Fitted(epoch)
    modal_names = ast.literal_eval(args.model_name)
    num_modalities = len(modal_names)

    all_acc_fusion, all_f1_fusion = [], []
    modal_acc_lists = [[] for _ in range(num_modalities)]
    modal_f1_lists = [[] for _ in range(num_modalities)]
    modal_loss_sums = [0.0 for _ in range(num_modalities)]

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs} [Training]")
    for step, data_packet in enumerate(pbar):
        # ========optimizer to zero_grad======
        if optimizer_m1 is not None:
            optimizer_m1.zero_grad()
        if optimizer_m2 is not None:
            optimizer_m2.zero_grad()
        if optimizer_m3 is not None:
            optimizer_m3.zero_grad()
        if optimizer_fusion is not None:
            optimizer_fusion.zero_grad()

        if modal_names == ["Visual", "Audio"]:
            spec, image,  label  = data_packet[0],data_packet[1],data_packet[2]
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
        outputs = model(data_mini_packet, epoch=0, labels=label, epoch_index=step)

        fusion_logits = outputs[0]
        modal_logits = list(outputs[1:1 + num_modalities])
        extra = list(outputs[1 + num_modalities:])

        masks = []
        if extra:
            if len(extra) == num_modalities:
                masks = extra
            elif len(extra) == num_modalities + 1:
                masks = extra[1:]
        masks = [m.float() if isinstance(m, torch.Tensor) else None for m in masks]

        acc_fusion, f1_fusion = calculate_metrics(fusion_logits, label)
        all_acc_fusion.append(acc_fusion)
        all_f1_fusion.append(f1_fusion)

        modal_losses_each = [criterion(logit, label) for logit in modal_logits]
        weighted_modal_losses = []
        modal_loss_values = []
        for idx, (loss_each, mask) in enumerate(zip(modal_losses_each, masks + [None] * (num_modalities - len(masks)))):
            # mask = None
            if mask is not None and args.Use_MACE:
                weight = mask.view(loss_each.size(0)).clamp_min(0)
                # print(f"Mask applied for {modal_names[idx]}: {weight.sum().item()} samples unmasked in batch of size {loss_each.size(0)}")
                weighted = (loss_each * weight).sum() / weight.sum().clamp_min(1e-6)
            else:
                weighted = loss_each.mean()
                # print(f"mean loss applied for {modal_names[idx]}")
            weighted_modal_losses.append(weighted)
            modal_loss_values.append(weighted.detach().item())

        for idx, logits in enumerate(modal_logits):
            acc_i, f1_i = calculate_metrics(logits, label)
            modal_acc_lists[idx].append(acc_i)
            modal_f1_lists[idx].append(f1_i)
            if idx < len(modal_loss_values):
                modal_loss_sums[idx] += modal_loss_values[idx]

        loss_fusion = criterion(fusion_logits, label).mean()
        if args.modality == 'full':
            loss_single = sum(weighted_modal_losses) * args.alpha
            loss = loss_fusion + loss_single
        else:
            target_name = args.modality.capitalize()
            if target_name not in modal_names:
                raise ValueError(f"Modal '{args.modality}' not found in {modal_names}")
            idx = modal_names.index(target_name)
            loss = weighted_modal_losses[idx] * args.alpha
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

        postfix = {
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{acc_fusion:.4f}',
            'F1': f'{f1_fusion:.4f}',
        }
        for name, acc_i, loss_val in zip(modal_names, [l[-1] for l in modal_acc_lists], modal_loss_values):
            postfix[f'Acc_{name}'] = f'{acc_i:.4f}'
            postfix[f'Loss_{name}'] = f'{loss_val:.4f}'
        if args.MaskType == "AME" and hasattr(model, "fusion_model"):
            for idx, name in enumerate(modal_names, start=1):
                counter = getattr(model.fusion_model, f"m{idx}_mask_num", None)
                if isinstance(counter, torch.Tensor):
                    postfix[f'{name}_mask_num'] = f'{counter.item():.1f}'
        pbar.set_postfix(postfix)

    avg_acc_fusion = sum(all_acc_fusion) / len(all_acc_fusion) if all_acc_fusion else 0.0
    avg_f1_fusion = sum(all_f1_fusion) / len(all_f1_fusion) if all_f1_fusion else 0.0

    modal_avg_metrics = {}
    for name, acc_list, f1_list in zip(modal_names, modal_acc_lists, modal_f1_lists):
        avg_acc = sum(acc_list) / len(acc_list) if acc_list else 0.0
        avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
        modal_avg_metrics[name] = (avg_acc, avg_f1)

    epoch_time = time.time() - epoch_start_time
    print(f"\nTrain Epoch {epoch} Summary (Average):")
    print(f"  Fusion -> Accuracy: {avg_acc_fusion:.4f}, F1-Score: {avg_f1_fusion:.4f}")
    for name in modal_names:
        acc, f1 = modal_avg_metrics.get(name, (0.0, 0.0))
        print(f"  {name} -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    print(f"  Epoch Time: {epoch_time:.2f}s ({epoch_time/60:.2f}min)")

    return (avg_acc_fusion, avg_f1_fusion), modal_avg_metrics, epoch_time

def valid(args, model, device, dataloader):
    with torch.no_grad():
        model.eval()

        modal_names = ast.literal_eval(args.model_name)
        num_modalities = len(modal_names)

        all_labels = []
        fusion_preds = []
        modal_preds = [[] for _ in range(num_modalities)]

        pbar = tqdm(dataloader, desc="Validating")
        for step, data_packet in enumerate(pbar):
            # _, batch = data_packet
            if modal_names == ["Visual", "Audio"]:
                spec, image,label  = data_packet[0],data_packet[1],data_packet[2]
                # spec, image, label, _ = batch
                spec, image, label = spec.to(device), image.to(device), label.to(device)
                if args.dataset == 'CREMAD':
                    data_mini_packet = (image.float(), spec.float())
                else:
                    data_mini_packet = (image.float(), spec.unsqueeze(1).float())
                outputs = model(data_mini_packet)
            elif modal_names == ["Image", "Text"]:
                token, padding_mask, image, label, _ = data_packet
                token, padding_mask = token.to(device), padding_mask.to(device)
                image, label = image.to(device), label.to(device)
                data_mini_packet = (token, padding_mask, image)
                outputs = model(data_mini_packet)
            elif modal_names == ["Text", "Visual", "Audio"]:
                token, padding_mask, image, spec, label, _ = data_packet
                token, padding_mask = token.to(device), padding_mask.to(device)
                image, spec, label = image.to(device), spec.to(device), label.to(device)
                data_mini_packet = (token, padding_mask, image.float(), spec.unsqueeze(1).float())
                outputs = model(data_mini_packet)
            else:
                raise NotImplementedError(f"Unsupported modal combination: {modal_names}")

            fusion_logits = outputs[0]
            modal_logits = list(outputs[1:1 + num_modalities])

            acc_fusion, f1_fusion = calculate_metrics(fusion_logits, label)
            postfix = {
                'Acc': f'{acc_fusion:.4f}',
                'F1': f'{f1_fusion:.4f}',
            }
            for name, logits in zip(modal_names, modal_logits):
                acc_modal, f1_modal = calculate_metrics(logits, label)
                postfix[f'Acc_{name}'] = f'{acc_modal:.4f}'
                postfix[f'F1_{name}'] = f'{f1_modal:.4f}'
            pbar.set_postfix(postfix)

            fusion_preds.append(fusion_logits.cpu())
            for idx, logits in enumerate(modal_logits):
                modal_preds[idx].append(logits.cpu())
            all_labels.append(label.cpu())

        fusion_preds = torch.cat(fusion_preds, dim=0)
        modal_preds = [torch.cat(preds, dim=0) for preds in modal_preds]
        all_labels = torch.cat(all_labels, dim=0)

        acc_fusion, f1_fusion = calculate_metrics(fusion_preds, all_labels)
        modal_metrics = {}
        for name, preds in zip(modal_names, modal_preds):
            acc_modal, f1_modal = calculate_metrics(preds, all_labels)
            modal_metrics[name] = (acc_modal, f1_modal)

        print(f"\nValidation Summary (Overall):")
        print(f"  Fusion -> Accuracy: {acc_fusion:.4f}, F1-Score: {f1_fusion:.4f}")
        for name in modal_names:
            acc_modal, f1_modal = modal_metrics.get(name, (0.0, 0.0))
            print(f"  {name} -> Accuracy: {acc_modal:.4f}, F1-Score: {f1_modal:.4f}")

    return (acc_fusion, f1_fusion), modal_metrics

def main():
    args=get_arguments()
    print(args)
    setup_seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ==================数据集加载=====================================
    if args.dataset == 'CREMAD':
        args.num_classes = 6
        # train_dataset = CramedDataset(mode='train', args=args)
        # test_dataset = CramedDataset(mode='test', args=args)
        train_dataset = AV_CD_Dataset(mode='train')
        test_dataset = AV_CD_Dataset(mode='test')
        print(f"use train AV_CD_Dataset and test AV_CD_Dataset for CREMAD dataset")
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
        train_dataset = M3AEDataset(args,mode='train')
        test_dataset = M3AEDataset(args,mode='test')
    elif args.dataset == 'MVSA':
        args.num_classes = 3
        train_dataset = M3AEDataset(args,mode='train')
        test_dataset = M3AEDataset(args,mode='test')
    elif args.dataset == 'IEMOCAP3':
        args.num_classes = 5
        train_dataset = TVADataset(mode='train', args=args, pick_num=3)
        test_dataset = TVADataset(mode='test', args=args, pick_num=3)
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support AVE, KineticSound and CREMA-D for now!'.format(args.dataset))
    print("train_dataset size: {}".format(len(train_dataset)))
    print("test_dataset size: {}".format(len(test_dataset)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    if args.model_name == '["Visual","Audio"]' or args.model_name == '["Image","Text"]':
        print(f"Using {args.model_name} model for training.")
        model = VA_Classifier(args)
    else:
        model = TVA_Classifier(args)
    # model = AVClassifier(args)
    print_model_params(model)
    model.to(device)
    if args.model_name == '["Visual","Audio"]' and args.Use_initWeight:
        model.apply(weight_init)
        print(f"Use Weight int")
        
    # ==================优化器和学习率调度器=============================
    optimizer_m1_params, optimizer_m2_params, optimizer_fusion_params = [], [], []
    optimizer_m1_name, optimizer_m2_name, optimizer_fusion_name =  [], [], []
    for name, param in model.named_parameters():
        if 'encoder_1' in name or 'm1' in name:
            optimizer_m1_params.append(param)
            optimizer_m1_name.append(name)
        elif 'encoder_2' in name or 'm2' in name:
            optimizer_m2_params.append(param)
            optimizer_m2_name.append(name)
        elif 'encoder_3' in name or 'm3' in name:
            if 'optimizer_m3_params' not in locals():
                optimizer_m3_params = []
                optimizer_m3_name = []
            optimizer_m3_params.append(param)
            optimizer_m3_name.append(name)
        else:
            optimizer_fusion_params.append(param)
            optimizer_fusion_name.append(name)

    # print("Optimizer 1 parameter names:", optimizer_m1_name)
    # print("Optimizer 2 parameter names:", optimizer_m2_name)
    # print("Fusion parameter names:", optimizer_fusion_name)
    optimizer_m1,optimizer_m2,optimizer_m3 = None, None, None
    if args.optimizer == "Adamw":
        if args.fusion_method == 'sum':
            optimizer_fusion = torch.optim.AdamW(model.parameters(), lr=args.learning_rate_fusion, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
        else:
            if args.model_name == '["Visual","Audio"]':
                # print(f"init optimizer")
                # optimizer_m1 = torch.optim.AdamW(optimizer_m1_params, lr=args.learning_rate_visual, betas=(0.9, 0.999),weight_decay=0.001,)
                # optimizer_m2 = torch.optim.AdamW(optimizer_m2_params, lr=args.learning_rate_audio, betas=(0.9, 0.999),weight_decay=0.001,)
                optimizer_m1 = torch.optim.AdamW(optimizer_m1_params, lr=args.learning_rate_visual, betas=(0.9, 0.999),weight_decay=args.weight_decay,)
                optimizer_m2 = torch.optim.AdamW(optimizer_m2_params, lr=args.learning_rate_audio, betas=(0.9, 0.999),weight_decay=args.weight_decay,)
            elif args.model_name == '["Image","Text"]':
                optimizer_m1 = torch.optim.AdamW(optimizer_m1_params, lr=args.learning_rate_image, betas=(0.9, 0.999),weight_decay=args.weight_decay,)
                optimizer_m2 = torch.optim.AdamW(optimizer_m2_params, lr=args.learning_rate_text, betas=(0.9, 0.999),weight_decay=args.weight_decay,)
            elif args.model_name == '["Text","Visual","Audio"]':
                # optimizer_m1 = torch.optim.AdamW(optimizer_m1_params, lr=args.learning_rate_text, betas=(0.9, 0.999),weight_decay=args.weight_decay,)
                optimizer_m1 = torch.optim.AdamW(optimizer_m1_params, lr=args.learning_rate_text, betas=(0.9, 0.999),weight_decay=0.01,)
                optimizer_m2 = torch.optim.AdamW(optimizer_m2_params, lr=args.learning_rate_visual, betas=(0.9, 0.999),weight_decay=args.weight_decay,)
                optimizer_m3 = torch.optim.AdamW(optimizer_m3_params, lr=args.learning_rate_audio, betas=(0.9, 0.999),weight_decay=args.weight_decay,)
            optimizer_fusion = torch.optim.AdamW(optimizer_fusion_params, lr=args.learning_rate_fusion, betas=(0.9, 0.999),weight_decay=args.weight_decay,)
            # optimizer_fusion = torch.optim.AdamW(optimizer_fusion_params, lr=args.learning_rate_fusion, betas=(0.9, 0.999),weight_decay=0.001,)
    elif args.optimizer == "adam":
        if args.fusion_method == 'sum':
            optimizer_fusion = torch.optim.Adam(model.parameters(), lr=args.learning_rate_fusion, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
        else:
            if args.model_name == '["Visual","Audio"]':
                optimizer_m1 = torch.optim.Adam(optimizer_m1_params, lr=args.learning_rate_visual, betas=(0.9, 0.999),weight_decay=args.weight_decay, eps=1e-8, amsgrad=False)
                optimizer_m2 = torch.optim.Adam(optimizer_m2_params, lr=args.learning_rate_audio, betas=(0.9, 0.999),weight_decay=args.weight_decay, eps=1e-8, amsgrad=False)
            elif args.model_name == '["Image","Text"]':
                optimizer_m1 = torch.optim.Adam(optimizer_m1_params, lr=args.learning_rate_image, betas=(0.9, 0.999),weight_decay=args.weight_decay, eps=1e-8, amsgrad=False)
                optimizer_m2 = torch.optim.Adam(optimizer_m2_params, lr=args.learning_rate_text, betas=(0.9, 0.999),weight_decay=args.weight_decay, eps=1e-8, amsgrad=False)
            elif args.model_name == '["Text","Visual","Audio"]':
                optimizer_m1 = torch.optim.Adam(optimizer_m1_params, lr=args.learning_rate_text, betas=(0.9, 0.999),weight_decay=args.weight_decay, eps=1e-8, amsgrad=False)
                optimizer_m2 = torch.optim.Adam(optimizer_m2_params, lr=args.learning_rate_visual, betas=(0.9, 0.999),weight_decay=args.weight_decay, eps=1e-8, amsgrad=False)
                optimizer_m3 = torch.optim.Adam(optimizer_m3_params, lr=args.learning_rate_audio, betas=(0.9, 0.999),weight_decay=args.weight_decay, eps=1e-8, amsgrad=False)
            if optimizer_fusion_params!=[]:
                optimizer_fusion = torch.optim.Adam(optimizer_fusion_params, lr=args.learning_rate_fusion, betas=(0.9, 0.999),weight_decay=args.weight_decay, eps=1e-8, amsgrad=False)
            else:
                optimizer_fusion = None
    elif args.optimizer == 'SGD':
        optimizer_fusion = torch.optim.SGD(model.parameters(), lr=args.learning_rate_fusion, momentum=0.9, weight_decay=1e-4)
    # optimizer_fusion = torch.optim.AdamW(model.parameters(), lr=args.learning_rate_fusion, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
    scheduler_map = None
    if args.fusion_method == 'sum' or args.optimizer == 'SGD':
        scheduler_fusion = torch.optim.lr_scheduler.StepLR(optimizer_fusion, args.lr_decay_step, args.lr_decay_ratio)
        scheduler_map = [scheduler_fusion]
    elif args.model_name == '["Text","Visual","Audio"]':
        scheduler_m1 = torch.optim.lr_scheduler.StepLR(optimizer_m1,args.lr_decay_step,args.lr_decay_ratio)
        scheduler_m2 = torch.optim.lr_scheduler.StepLR(optimizer_m2,args.lr_decay_step,args.lr_decay_ratio)
        scheduler_m3 = torch.optim.lr_scheduler.StepLR(optimizer_m3,args.lr_decay_step,args.lr_decay_ratio)
        if optimizer_fusion is not None:
            scheduler_fusion = torch.optim.lr_scheduler.StepLR(optimizer_fusion,args.lr_decay_step,args.lr_decay_ratio)
            scheduler_map  =[scheduler_m1,scheduler_m2,scheduler_m3,scheduler_fusion]
        else:
            scheduler_map  =[scheduler_m1,scheduler_m2,scheduler_m3]
    else:
        scheduler_m1 = torch.optim.lr_scheduler.StepLR(optimizer_m1,args.lr_decay_step,args.lr_decay_ratio)
        scheduler_m2 = torch.optim.lr_scheduler.StepLR(optimizer_m2,args.lr_decay_step,args.lr_decay_ratio)
        if optimizer_fusion is not None:
            scheduler_fusion = torch.optim.lr_scheduler.StepLR(optimizer_fusion,args.lr_decay_step,args.lr_decay_ratio)
            scheduler_map  =[scheduler_m1,scheduler_m2,scheduler_fusion]
        else:
            scheduler_map  =[scheduler_m1,scheduler_m2]
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate_text, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)
    # optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate_fusion, betas=(0.9, 0.999),weight_decay=0.01,)

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    log_path = os.path.join(args.ckpt_path, args.dataset + '_' + args.fusion_method + '.csv')
    modal_names = ast.literal_eval(args.model_name)
    modal_header = []
    for name in modal_names:
        modal_header.extend([f"Val_Acc_{name}", f"Val_F1_{name}"])
    with open(log_path, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(['Epoch', 'Val_Acc', 'Val_F1', *modal_header])
    # ==================训练和验证=====================================
    if args.train:
        best_acc = 0.0
        save_path = None
        val_metrics = {'acc':-1,'f1':-1,'acc_a':-1,'f1_a':-1,'acc_v':-1,'f1_v':-1}
        epoch_times = []  # 记录每个epoch的总时间（remix+training）
        remix_times = []  # 记录每个epoch的remix时间
        
        # Remix: 用于存储模态特定数据集
        remix_datasets = None
        
        for epoch in range(args.current_epoch, args.epochs + 1):
            print(f"\n=== Epoch: {epoch}/{args.epochs} ===")
            print_current_lrs(optimizer_m1, optimizer_m2, optimizer_fusion, optimizer_m3, scheduler_map)
            
            epoch_total_start = time.time()  # 记录整个epoch的开始时间（不包括validation）
            
            # 判断是否使用 Remix 训练
            # if args.use_remix and epoch > args.remix_warmup and epoch%2==1:
            if args.use_remix and epoch > args.remix_warmup:
                # Remix 阶段：先进行 remix 采样，然后模态特定训练
                # if remix_datasets is None or epoch == args.remix_warmup + 1:
                #     # 第一次进入 Remix 或每隔一定周期重新采样
                #     remix_datasets = remix(args, model, device, train_dataloader, epoch)
                remix_datasets, remix_time = remix(args, model, device, train_dataloader, epoch)
                remix_times.append(remix_time)
                # 使用模态特定训练
                batch_loss, train_time = special_train_epoch(
                    args=args,
                    epoch=epoch,
                    model=model,
                    device=device,
                    modality_datasets=remix_datasets,
                    optimizer_m1=optimizer_m1,
                    optimizer_m2=optimizer_m2,
                    optimizer_fusion=optimizer_fusion,
                    optimizer_m3=optimizer_m3
                )
            else:
                # Warmup 阶段：正常训练
                remix_time = 0.0  # warmup阶段没有remix
                remix_times.append(remix_time)
                train_fusion_metrics, train_modal_metrics, train_time = train_epoch(
                    args=args,
                    epoch=epoch,
                    model=model,
                    device=device,
                    dataloader=train_dataloader,
                    optimizer_m1=optimizer_m1,
                    optimizer_m2=optimizer_m2,
                    optimizer_fusion=optimizer_fusion,
                    optimizer_m3=optimizer_m3,
                    scheduler_map=scheduler_map, 
                )
            
            # 计算完整的epoch时间（remix + training，不包括validation）
            epoch_total_time = time.time() - epoch_total_start
            epoch_times.append(epoch_total_time)
            
            # 输出epoch总时间统计（在validation之前）
            print(f"\nEpoch {epoch} Total Time Summary (Training Phase):")
            if remix_times[-1] > 0:
                print(f"  Remix Time: {remix_times[-1]:.2f}s ({remix_times[-1]/60:.2f}min)")
                print(f"  Training Time: {train_time:.2f}s ({train_time/60:.2f}min)")
            else:
                print(f"  Training Time: {train_time:.2f}s ({train_time/60:.2f}min)")
            print(f"  Total Epoch Time: {epoch_total_time:.2f}s ({epoch_total_time/60:.2f}min)")
            
            if scheduler_map != None:
                for sch in scheduler_map:
                    sch.step()
                
            val_fusion_metrics, val_modal_metrics = valid(args, model, device, test_dataloader)

            if args.use_tensorboard:
                writer = SummaryWriter(log_dir=args.tensorboard_path)
                writer.add_scalar('Train/Accuracy', train_fusion_metrics[0], epoch)
                writer.add_scalar('Train/F1_Score', train_fusion_metrics[1], epoch)
                writer.add_scalar('Val/Accuracy', val_fusion_metrics[0], epoch)
                writer.add_scalar('Val/F1_Score', val_fusion_metrics[1], epoch)
                writer.close()

            is_best = val_fusion_metrics[0] > best_acc
            best_acc = max(val_fusion_metrics[0], best_acc)
            print(f"Best Val Acc: {best_acc:.4f}")

            if is_best:
                save_path = os.path.join(
                    args.ckpt_path,
                    f"{epoch}_{int(best_acc * 1000)}_"
                    f"{args.dataset}_{args.fusion_method}_"
                    f"alpha{args.alpha}_gamma{args.ame_gama}_beta{args.ame_beta}_"
                    f"Mask-{args.MaskType}_"
                    f"dim{args.unified_dim}_"
                    f"M1Tok{args.m1_token_len}_M2Tok{args.m2_token_len}_"
                    f"bs{args.batch_size}_"
                    f"{datetime.datetime.now().month}_{datetime.datetime.now().day}.pth"
                )
                torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
                val_metrics = {'Fusion': val_fusion_metrics, **val_modal_metrics}

            with open(log_path, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=",")
                row = [epoch, val_fusion_metrics[0], val_fusion_metrics[1]]
                for name in modal_names:
                    acc_modal, f1_modal = val_modal_metrics.get(name, (0.0, 0.0))
                    row.extend([acc_modal, f1_modal])
                writer.writerow(row)
        
        # 输出训练时间统计
        print(f"\n{'='*50}")
        print(f"Training Time Statistics (Excluding Validation)")
        print(f"{'='*50}")
        if epoch_times:
            total_time = sum(epoch_times)
            total_remix_time = sum(remix_times)
            total_train_time = total_time - total_remix_time
            avg_time = total_time / len(epoch_times)
            avg_remix_time = total_remix_time / len(remix_times) if remix_times else 0
            min_time = min(epoch_times)
            max_time = max(epoch_times)
            
            print(f"Total Time (All Epochs): {total_time:.2f}s ({total_time/60:.2f}min / {total_time/3600:.2f}h)")
            print(f"  - Total Remix Time: {total_remix_time:.2f}s ({total_remix_time/60:.2f}min / {total_remix_time/3600:.2f}h)")
            print(f"  - Total Training Time: {total_train_time:.2f}s ({total_train_time/60:.2f}min / {total_train_time/3600:.2f}h)")
            print(f"\nAverage Time per Epoch: {avg_time:.2f}s ({avg_time/60:.2f}min)")
            if avg_remix_time > 0:
                print(f"  - Average Remix Time: {avg_remix_time:.2f}s ({avg_remix_time/60:.2f}min)")
            print(f"Min Epoch Time: {min_time:.2f}s ({min_time/60:.2f}min)")
            print(f"Max Epoch Time: {max_time:.2f}s ({max_time/60:.2f}min)")
            
            print(f"\nEpoch-wise Time Breakdown:")
            for i, (total_t, remix_t) in enumerate(zip(epoch_times, remix_times), start=args.current_epoch):
                train_t = total_t - remix_t
                if remix_t > 0:
                    print(f"  Epoch {i}: {total_t:.2f}s ({total_t/60:.2f}min) [Remix: {remix_t:.2f}s, Train: {train_t:.2f}s]")
                else:
                    print(f"  Epoch {i}: {total_t:.2f}s ({total_t/60:.2f}min) [Train only]")
        print(f"{'='*50}\n")
        
        with open(f"Results/results-Remix-AME-{args.dataset}-{args.fusion_method}.log", "a") as f:
            f.write(
                f"==================== {datetime.datetime.now()} ===================\n \n"
            )
            f.write(f"========================={args.model_save_name}==================================\n")
            f.write(f"val_acc: {best_acc}\n")
            f.write(f"all metric: {val_metrics}\n")
            f.write(f"best model save as {save_path}\n")
            if epoch_times:
                total_time = sum(epoch_times)
                total_remix_time = sum(remix_times)
                total_train_time = total_time - total_remix_time
                avg_time = total_time / len(epoch_times)
                f.write(f"total_time: {total_time:.2f}s ({total_time/60:.2f}min / {total_time/3600:.2f}h)\n")
                f.write(f"total_remix_time: {total_remix_time:.2f}s ({total_remix_time/60:.2f}min / {total_remix_time/3600:.2f}h)\n")
                f.write(f"total_training_time: {total_train_time:.2f}s ({total_train_time/60:.2f}min / {total_train_time/3600:.2f}h)\n")
                f.write(f"avg_epoch_time: {avg_time:.2f}s ({avg_time/60:.2f}min)\n")
            f.write(f"args: {args}\n \n")
        print(f"val best metrics is {val_metrics} \n")
        print(f"best model save as {save_path} \n")
        print(f"args:{args} \n \n")
    # scheduler_m1 = get_scheduler(optimizer_m1, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    # scheduler_m2 = get_scheduler(optimizer_m2, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    # scheduler_fusion = get_scheduler(optimizer_fusion, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
if __name__ == '__main__':
    main()