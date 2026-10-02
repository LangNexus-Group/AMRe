# 这个文件是原始的, 仅进行训练操作, 不会额外记录相关参数，通过设置参数进行消融实验
import os
import csv
import ast
import time
import torch
import random
import datetime
import argparse

# import soap as SOAP

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
from dataset.Mydataset import TVADataset
from dataset.Mydataset import M3AEDataset
from dataset.Mydataset import CramedDataset,AVEDataset,KSDataset

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
    parser.add_argument('--learning_rate_visual', default=1e-3, type=float, help='Learning rate for visual encoder')
    parser.add_argument('--learning_rate_audio', default=1e-3, type=float, help='Learning rate for audio encoder')
    parser.add_argument('--learning_rate_text', default=1e-5, type=float, help='Learning rate for text encoder')
    parser.add_argument('--learning_rate_image', default=1e-3, type=float, help='Learning rate for text encoder')
    parser.add_argument('--learning_rate_fusion', default=1e-3, type=float, help='Learning rate for fusion model')
    parser.add_argument('--optimizer', default='Adamw', type=str,help='Optimizer to use (SGD, Adam, Adamw,SOAP)')
    parser.add_argument('--lr_decay_step',default=30, type=int)
    parser.add_argument('--lr_decay_ratio',default=0.1, type=float)
    parser.add_argument('--weight_decay',default=1e-4, type=float)
    # 基础设置
    parser.add_argument('--random_seed', default=2024, type=int)
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--gpu_ids', default='0', type=str, help='GPU ids')
    parser.add_argument('--current_epoch', type=int, default=1,help="Start train epoch number")
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs')
    parser.add_argument('--Use_initWeight', default=False, type=bool, help='Use weight init model')
    parser.add_argument('--mask_percent', default=0, type=float, help='mask data percent')
    

    # 模型设置
    parser.add_argument('--fusion_method', default='concat', type=str,choices=['sum', 'concat', 'Gate', 'Film', 'MMTM', 'CA', 'CentralNet'], help='Fusion method to combine modalities')
    parser.add_argument('--model_name', default='["Visual","Audio"]', type=str, choices=['["Visual","Audio"]', '["Image","Text"]', '["Text","Visual","Audio"]'])
    parser.add_argument('--modality',default='full',type=str,choices=['full','Audio','Visual','Image','Text'],help='modality to use')
    parser.add_argument('--unified_dim', default=512, type=int, help='Unified feature dimension after encoders')
    parser.add_argument('--m1_token_len', default=1, type=int, help='Modality 1 (e.g., visual) token length')
    parser.add_argument('--m2_token_len', default=1, type=int, help='Modality 2 (e.g., audio or text) token length')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes')
    parser.add_argument('--m1_gate', default=False, type=bool, help='For Gate Model, whether to use modality 1 gate')
    parser.add_argument('--x_film', default=False, type=bool, help='For Film Model, whether to use modality 1 film')

    # AME相关参数
    parser.add_argument('--MaskType', default='None', type=str, choices=['None', 'AME','Shapley','AME_attention','AME_DynamicBetaSoftGate','Random'], help='Type of masking strategy')
    parser.add_argument('--alpha', default=1.0, type=float, help='Alpha parameter for AME module')
    parser.add_argument('--ame_gama', default=0.1, type=float, help='Gamma parameter for AME module')
    parser.add_argument('--ame_gap', default=2, type=int, help='restoration gap')
    parser.add_argument('--ame_gap_start', default=1, type=int, help='restoration gap start')
    parser.add_argument('--ame_beta', default=0.7, type=float, help='Beta parameter for AME module')
    parser.add_argument('--ame_temperature', default=0.2, type=float, help='Temperature parameter for AME module')
    parser.add_argument('--warmup_epoch', default=0, type=int, help='Number of warmup epochs before applying AME')
    parser.add_argument('--Use_MACE',default=True, type=bool, help="是否使用MACE loss")
    parser.add_argument('--random_mask_drop_prob',default=0.3, type=float, help="随机掩码的比例")
    parser.add_argument('--ame_acc_metric',default='ce_loss', type=str, help="准确性的评价指标")
    parser.add_argument('--ame_unc_metric',default='kl', type=str, help="确定性的评价指标")
    
    
    # tensorboard相关参数
    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')
    # 保存模型的相关参数
    parser.add_argument('--ckpt_path', required=True, type=str, help='path to save trained models')
    parser.add_argument("--model_save_name",required=True,type=str,help='model save log name')
    parser.add_argument('--pretrained_model_path', default='', type=str, help='Path to a pretrained model for evaluation or fine-tuning')
    # OGM-GE 模型相关参数
    parser.add_argument('--modulation', default='OGM_GE', type=str, choices=['OGM', 'OGM_GE'], help='Modulation strategy to use')
    parser.add_argument('--ogm_alpha', default=0.8, type=float, help='Alpha parameter for OGM modulation')
    parser.add_argument('--modulation_starts', default=0, type=int, help='Epoch to start applying OGM modulation')
    parser.add_argument('--modulation_ends', default=50, type=int, help='Epoch to stop applying OGM modulation')
    parser.add_argument('--Use_OGM', default=False, type=bool, help='Whether to use OGM modulation')
    
    # LFM相关参数
    parser.add_argument('--LFM', default=False, type=bool, help='Whether to use LFM method')
    parser.add_argument('--unimodal_use', default=None, type=str, help='Use unimodal train')
    
    
    args = parser.parse_args()
    return args


def train_epoch(args, epoch, model, device, dataloader, optimizer_m1, optimizer_m2, optimizer_fusion, optimizer_m3=None,scheduler_map=None):
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
        if optimizer_m1 is not None:
            optimizer_m1.zero_grad()
        if optimizer_m2 is not None:
            optimizer_m2.zero_grad()
        if optimizer_m3 is not None:
            optimizer_m3.zero_grad()
        if optimizer_fusion is not None:
            optimizer_fusion.zero_grad()

        if modal_names == ["Visual", "Audio"]:
            # image, spec, label, _, _ = data_packet
            spec, image, label, sid = data_packet[0],data_packet[1],data_packet[2], data_packet[-1]
            spec, image, label = spec.to(device), image.to(device), label.to(device)
            if args.dataset == 'CREMAD':
                data_mini_packet = (image.float(), spec.float())
            else:
                data_mini_packet = (image.float(), spec.unsqueeze(1).float())
            # data_mini_packet = (image.float(), spec.float())
            # outputs = model(data_mini_packet, epoch=epoch, labels=label, epoch_index=step)
        elif modal_names == ["Image", "Text"]:
            token, padding_mask, image, label, sid = data_packet
            token, padding_mask = token.to(device), padding_mask.to(device)
            image, label = image.to(device), label.to(device)
            data_mini_packet = (token, padding_mask, image)
            # outputs = model(data_mini_packet, epoch=epoch, labels=label, epoch_index=step)
        elif modal_names == ["Text", "Visual", "Audio"]:
            token, padding_mask, image, spec, label, sid = data_packet
            token, padding_mask = token.to(device), padding_mask.to(device)
            image, spec, label = image.to(device), spec.to(device), label.to(device)
            # sid = sid.to(device)
            data_mini_packet = (token, padding_mask, image.float(), spec.unsqueeze(1).float())
            # outputs = model(data_mini_packet, epoch=epoch, labels=label, epoch_index=step)
        else:
            raise NotImplementedError(f"Unsupported modal combination: {modal_names}")
        # # print(f"sid is {sid}")
        # # sid = torch.tensor(sid, dtype=torch.long).to(device) if not isinstance(sid, torch.Tensor) else sid.to(device)
        # # 仅作为标识符，不参与计算时的简化处理
        # if not isinstance(sid, torch.Tensor):
        #     # 转为字符串张量（无需编码，仅存储标识符）
        #     sid = torch.tensor(sid, dtype=torch.string_).to(device)
        # else:
        #     sid = sid.to(device)
        outputs = model(data_mini_packet, epoch=epoch, labels=label, epoch_index=step, sid=sid)
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
        if cls_k is not None:
            # print(f"使用LFM")
            if num_modalities == 2:
                loss_alignment = Alignment(outputs[1],outputs[2])
                loss_cls = criterion(0.5*outputs[1] + 0.5*outputs[2],label).mean()
                # print(f"使用LFM")
                
            elif num_modalities == 3:
                loss_alignment = (Alignment(outputs[1],outputs[2]) + Alignment(outputs[1],outputs[3]) + Alignment(outputs[2],outputs[3]))/num_modalities
                loss_cls = criterion((outputs[1]+outputs[2]+outputs[3])/3, label).mean() # average over all modalities
            # print(cls_k)
            # loss_single = sum(weighted_modal_losses) * args.alpha
            loss = 2 * (cls_k[0] * loss_cls + cls_k[1] * loss_alignment)
            # loss = 2 * (cls_k[0] * loss_cls + cls_k[1] * loss_alignment)+loss_single
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)

        # OGM-GE 代码实现 (支持任意数量模态)
        def apply_modulation(outputs, epoch):
            """
            OGM 梯度调制函数 - 支持任意数量模态
            
            核心思想：
            1. 计算每个模态的 score (正确类别的概率之和)
            2. 计算每个模态相对于其他模态的 ratio
            3. 识别最强模态（ratio 最大）
            4. 对最强模态施加梯度抑制系数
            
            Args:
                outputs: 模型输出 (fusion_logits, modal_1_logits, modal_2_logits, ...)
                epoch: 当前 epoch
            """
            # 1. 计算每个模态的 score
            scores = []
            for modal_idx in range(num_modalities):
                modal_logits = outputs[modal_idx + 1]  # outputs[0] 是 fusion
                # score = sum of P(correct_class) for all samples
                score = sum([F.softmax(modal_logits, dim=1)[i][label[i]] 
                            for i in range(modal_logits.size(0))])
                scores.append(score)
            
            # 2. 计算每个模态相对于其他模态的 ratio
            ratios = []
            for i in range(num_modalities):
                # ratio_i = score_i / sum(score_j for j != i)
                other_scores_sum = sum(scores[j] for j in range(num_modalities) if j != i)
                ratio = scores[i] / (other_scores_sum + 1e-8)  # 防止除零
                ratios.append(ratio)
            
            # 3. 找到最强模态 (ratio 最大)
            max_ratio_idx = ratios.index(max(ratios))
            max_ratio = ratios[max_ratio_idx]
            
            # 4. 计算调制系数
            # 最强模态: coeff = 1 - tanh(alpha * relu(ratio))
            # 其他模态: coeff = 1 (不调制)
            coeffs = []
            for i in range(num_modalities):
                if i == max_ratio_idx:
                    coeff = 1 - torch.tanh(args.ogm_alpha * torch.relu(ratios[i]))
                else:
                    coeff = torch.tensor(1.0).to(device)
                coeffs.append(coeff)
            
            # 5. TensorBoard 记录
            if args.use_tensorboard:
                iteration = epoch * len(dataloader) + step
                for i in range(num_modalities):
                    csv.writer.add_scalar(f'OGM/ratio_m{i+1}', ratios[i], iteration)
                    csv.writer.add_scalar(f'OGM/coeff_m{i+1}', coeffs[i], iteration)
                csv.writer.add_scalar('OGM/max_ratio_modality', max_ratio_idx + 1, iteration)
            
            # 6. 应用梯度调制
            if args.modulation_starts <= epoch <= args.modulation_ends:
                for name, parms in model.named_parameters():
                    if parms.grad is None:
                        continue
                    
                    # 只调制 4D 张量 (卷积层参数)
                    if len(parms.grad.size()) != 4:
                        continue
                    
                    # 检查参数属于哪个模态
                    # layer = str(name).split('.')[1] if '.' in name else name
                    # print(f"layer is {name}")
                    for modal_idx in range(num_modalities):
                        modal_tag = f'encoder_{modal_idx + 1}'
                        # print(f"modal_tag{modal_tag}")
                        if modal_tag in name:
                            coeff = coeffs[modal_idx]
                            if args.modulation == 'OGM_GE':
                                # OGM-GE: 添加高斯噪声
                                # print("开始调制")
                                
                                parms.grad = parms.grad * coeff + \
                                    torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                            elif args.modulation == 'OGM':
                                # OGM: 仅调制
                                parms.grad *= coeff
                            break  # 找到匹配的模态，跳出循环
        
        # 调用梯度调制函数
        if args.Use_OGM:
            # print(f"使用OGM")
            apply_modulation(outputs, epoch)

    
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
        if args.MaskType != "None" and hasattr(model, "fusion_model"):
            for idx, name in enumerate(modal_names, start=1):
                counter = getattr(model.fusion_model, f"m{idx}_mask_num", None)
                if isinstance(counter, torch.Tensor):
                    postfix[f'{name}_mask_num'] = f'{counter.item():.1f}'
        pbar.set_postfix(postfix)
    MaskNumber = {f"{name}_mask_number":0 for name in modal_names}
    for idx, name in enumerate(modal_names, start=1):
        mask_number = getattr(model.fusion_model, f"m{idx}_mask_num", None)
        MaskNumber[f"{name}_mask_number"] = mask_number.item()
    avg_acc_fusion = sum(all_acc_fusion) / len(all_acc_fusion) if all_acc_fusion else 0.0
    avg_f1_fusion = sum(all_f1_fusion) / len(all_f1_fusion) if all_f1_fusion else 0.0

    modal_avg_metrics = {}
    for name, acc_list, f1_list in zip(modal_names, modal_acc_lists, modal_f1_lists):
        avg_acc = sum(acc_list) / len(acc_list) if acc_list else 0.0
        avg_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
        modal_avg_metrics[name] = (avg_acc, avg_f1)

    print(f"\nTrain Epoch {epoch} Summary (Average):")
    print(f"  Fusion -> Accuracy: {avg_acc_fusion:.4f}, F1-Score: {avg_f1_fusion:.4f}")
    for name in modal_names:
        acc, f1 = modal_avg_metrics.get(name, (0.0, 0.0))
        print(f"  {name} -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")

    return (avg_acc_fusion, avg_f1_fusion), modal_avg_metrics, MaskNumber

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
    
    # ==========加载模型==========
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
    optimizer_m1_params, optimizer_m2_params, optimizer_fusion_params, optimizer_m3_params = [], [], [], []
    optimizer_m1_name, optimizer_m2_name, optimizer_fusion_name, optimizer_m3_name =  [], [], [], []
    for name, param in model.named_parameters():
        if 'encoder_1' in name or 'm1' in name:
            optimizer_m1_params.append(param)
            optimizer_m1_name.append(name)
        elif 'encoder_2' in name or 'm2' in name:
            optimizer_m2_params.append(param)
            optimizer_m2_name.append(name)
        elif 'encoder_3' in name or 'm3' in name:
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
                optimizer_m1 = torch.optim.AdamW(optimizer_m1_params, lr=args.learning_rate_text, betas=(0.9, 0.999),weight_decay=args.weight_decay,)
                # optimizer_m1 = torch.optim.AdamW(optimizer_m1_params, lr=args.learning_rate_text, betas=(0.9, 0.999),weight_decay=0.01,)
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
        modal_header.extend([f"Val_Acc_{name}", f"Val_F1_{name}",f"{name}_mask_number"])
    with open(log_path, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(['Epoch', 'Val_Acc', 'Val_F1', *modal_header])
    # ==================训练和验证=====================================
    if args.train:
        best_acc = 0.0
        save_path = None
        val_metrics = {'acc':-1,'f1':-1,'acc_a':-1,'f1_a':-1,'acc_v':-1,'f1_v':-1}
        for epoch in range(args.current_epoch, args.epochs + 1):
            print(f"\n=== Epoch: {epoch}/{args.epochs} ===")
            print_current_lrs(optimizer_m1, optimizer_m2, optimizer_fusion, optimizer_m3, scheduler_map)
            train_fusion_metrics, train_modal_metrics, Mask_Number = train_epoch(
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
                    mask_number = Mask_Number[f"{name}_mask_number"]
                    row.extend([acc_modal, f1_modal, mask_number])
                writer.writerow(row)
    else:
        loaded_dict = torch.load(args.pretrained_model_path, map_location=device)
        model.load_state_dict(loaded_dict)
        print(f"Loaded pretrained model from {args.pretrained_model_path}")
        val_fusion_metrics, val_modal_metrics = valid(args, model, device, test_dataloader)
        best_acc = val_fusion_metrics[0]
        val_metrics = {'Fusion': val_fusion_metrics, **val_modal_metrics}
        save_path = args.pretrained_model_path
    # with open(f"Results/results-AME-{args.dataset}-{args.fusion_method}.log", "a") as f:
    with open(f"Results/results-AMRe-{args.fusion_method}.log", "a") as f:
        f.write(
            f"==================== {datetime.datetime.now()} ===================\n \n"
        )
        f.write(f"========================={args.model_save_name}==================================\n")
        f.write(f"val_acc: {best_acc}\n")
        f.write(f"all metric: {val_metrics}\n")
        f.write(f"best model save as {save_path}\n")
        f.write(f"args: {args}\n \n")
    print(f"val best metrics is {val_metrics} \n")
    print(f"best model save as {save_path} \n")
    print(f"args:{args} \n \n")
    # scheduler_m1 = get_scheduler(optimizer_m1, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    # scheduler_m2 = get_scheduler(optimizer_m2, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    # scheduler_fusion = get_scheduler(optimizer_fusion, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
if __name__ == '__main__':
    main()