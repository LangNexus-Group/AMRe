
# © 2025 Xiaoyu Ma. 
# Code of Improving Multimodal Learning Balance and Sufficiency through Data Remixing.
# This code is adapted from OGM-GE, available at:
# https://github.com/GeWu-Lab/OGM-GE_CVPR2022
# All rights reserved.

import os
import csv
import glob
import time
import torch
import random
import librosa
import pickle
import csv 
import numpy as np
import torch.nn as nn
from PIL import Image
from numpy.random import randint
from torchvision import transforms
from torch.utils.data import Dataset
from timm.data import create_transform
from sklearn.preprocessing import OneHotEncoder
class CramedDataset_Remix(Dataset):
    """CREMA-D数据集的Remix版本，支持模态特定的训练"""
    
    def __init__(self, args, modality='audio'):
        """
        Args:
            args: 参数配置
            modality: 模态名称，如 'audio', 'visual' 等
        """
        self.args = args
        self.data = []
        self.mode = 'train'  # Remix始终用于训练
        self.modality = modality
        classes = []
        data2class = {}
        
        # 数据集路径配置
        self.data_root = "/root/autodl-tmp/AME/data/dataset/CREMA-D/NewData"
        self.visual_path = f'{self.data_root}/Image-01-FPS/'
        self.audio_path = f'{self.data_root}/audio/'
        self.stat_path = f'{self.data_root}/stat.csv'
        # self.train_txt = os.path.join(self.data_root, 'trainSet.txt')
        # self.test_txt = os.path.join(self.data_root, 'testSet.txt')
        
        # 根据模态读取对应的CSV文件
        if modality == 'audio':
            csv_file = './data/CREMAD/remix_audio.csv'
        elif modality == 'visual':
            csv_file = './data/CREMAD/remix_visual.csv'
        else:
            csv_file = f'./data/CREMAD/remix_{modality}.csv'
        
        # 读取类别信息
        with open(self.stat_path, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])
        self.classes = sorted(classes)
        
        # self.data2class = data2class
        # 读取模态特定的数据
        itmes = 0
        with open(csv_file, encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                if len(item) < 2:
                    continue
                file_name = item[0]
                label_name = self.classes[int(item[1])]
                
                # 验证文件存在
                audio_path = os.path.join(self.audio_path, file_name + '.npy')
                # visual_path = os.path.join(self.visual_path, file_name)
                visual_path = self.visual_path+file_name
                # if item[1] in classes and os.path.exists(self.audio_path + item[0] + '.npy') and os.path.exists(
                # self.visual_path + item[0]):
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.data.append(file_name)
                    data2class[file_name] = label_name

        self.data2class = data2class
        self.fps = getattr(args, 'fps', 2)  # 默认2帧，可通过args配置
        
        # print(f'{modality} data load over')
        # print(f'# of files = {len(self.data)}')
        # print(f'# of classes = {len(self.classes)}')

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Returns:
            tuple: (spectrogram, images, label, modality, sid)
                - spectrogram: 音频特征 [1, freq, time]
                - images: 视觉帧 [3, fps, 224, 224]
                - label: 类别索引
                - modality: 模态名称字符串
                - sid: 样本ID
        """
        datum = self.data[idx]

        # 音频处理
        spectrogram = np.load(os.path.join(self.audio_path, datum + '.npy'))
        spectrogram = np.expand_dims(spectrogram, axis=0)

        # 视觉处理 - 训练模式的transform
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
       
        folder_path = os.path.join(self.visual_path, datum)
        
        # 获取所有图像文件并排序
        image_samples = os.listdir(folder_path)
        image_samples.sort()
        
        # 随机选择fps帧
        if len(image_samples) < self.fps:
            select_index = np.random.choice(len(image_samples), size=self.fps, replace=True)
        else:
            select_index = np.random.choice(len(image_samples), size=self.fps, replace=False)
        select_index.sort()

        # 加载并处理选中的帧
        images = torch.zeros((self.fps, 3, 224, 224), dtype=torch.float32)
        for i in range(self.fps):
            img_path = os.path.join(folder_path, image_samples[select_index[i]])
            img = Image.open(img_path).convert('RGB')
            images[i] = transform(img)

        # 调整维度顺序：[fps, 3, 224, 224] -> [3, fps, 224, 224]
        images = torch.permute(images, (1, 0, 2, 3))
        
        # 获取标签
        label = self.classes.index(self.data2class[datum])
        
        # 返回格式：(spectrogram, images, label, modality, sid)
        # 与 AVEDataset_Remix 保持一致的返回顺序
        return spectrogram, images, label, self.modality, datum


class AVEDataset_Remix(Dataset):
    """AVE数据集的Remix版本，支持模态特定的训练"""
    
    def __init__(self, args, modality='audio'):
        """
        Args:
            args: 参数配置
            modality: 模态名称，如 'audio', 'visual' 等
        """
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = 'train'  # Remix始终用于训练
        self.modality = modality
        classes = []
        
        self.data_root = '/root/autodl-tmp/AME/data/dataset/AVE/AVE_Dataset'
        self.visual_feature_path = os.path.join(self.data_root, 'Image-01-FPS-SE')
        self.audio_feature_path = os.path.join(self.data_root, 'Audio-1004-SE')
        self.stat_path = os.path.join(self.data_root, 'testSet.txt')
        
        # 根据模态读取对应的CSV文件
        csv_file = f'./data/AVE/remix_{modality}.csv'
        
        # 读取类别信息
        with open(self.stat_path, 'r') as f1:
            files = f1.readlines()
            for item in files:
                item = item.split('&')
                if item[0] not in classes:
                    classes.append(item[0])
        class_dict = {}
        for i, c in enumerate(classes):
            class_dict[c] = i
        # print("class_dict:",class_dict)
        # 读取模态特定的数据
        with open(csv_file, 'r') as f2:
            csv_reader = csv.reader(f2)
            for row in csv_reader:
                if len(row) < 2:
                    continue
                audio_name = row[0]
                # label_name = row[1]
                label_value = row[1]
                
                # 验证文件存在
                audio_path = os.path.join(self.audio_feature_path, audio_name + '.pkl')
                visual_path = os.path.join(self.visual_feature_path, audio_name)
                # print(f"audio_path is {audio_path}")
                # print(f"visual_path is {visual_path}")
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    # self.label.append(class_dict[label_name])
                    
                    # 处理标签：如果是数字字符串则直接转换，否则通过class_dict查找
                    try:
                        label_idx = int(label_value)
                    except ValueError:
                        label_idx = class_dict.get(label_value, 0)
                    self.label.append(label_idx)
        
        # print(f'{modality} data load over')
        # print(f'# of files = {len(self.image)}')
        # print(f'# of classes = {len(classes)}')
    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        # 音频处理
        spectrogram = pickle.load(open(self.audio[idx], 'rb'))
        
        # 训练模式的transform
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 视觉处理
        image_samples = os.listdir(self.image[idx])
        image_samples.sort()
        
        if len(image_samples) < self.args.fps:
            select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=True)
        else:
            select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=False)
        select_index.sort()
        
        images = torch.zeros((self.args.fps, 3, 224, 224), dtype=torch.float32)
        for i in range(self.args.fps):
            img = Image.open(os.path.join(self.image[idx], image_samples[select_index[i]])).convert('RGB')
            img = transform(img)
            images[i] = img
        
        images = torch.permute(images, (1, 0, 2, 3))
        label = self.label[idx]
        sid = self.audio[idx].split('/')[-1].split('.')[0]
        
        return spectrogram, images, label, self.modality, sid


class M3AEDataset_Remix(Dataset):
    """M3AE数据集（Food101/MVSA）的Remix版本，支持文本和视觉模态特定的训练"""
    
    def __init__(self, args, modality='text'):
        """
        Args:
            args: 参数配置
            modality: 模态名称，如 'text', 'visual' 等
        """
        self.args = args
        self.mode = 'train'
        self.dataset = args.dataset
        self.modality = modality
        
        # 路径配置
        if args.dataset == "Food101":
            self.data_root = '/root/autodl-tmp/AME/data/dataset/Food101/Food101'
        elif args.dataset == "MVSA":
            self.data_root = '/root/autodl-tmp/AME/data/dataset/MVSA/MVSA_Single'
        
        self.visual_feature_path = os.path.join(self.data_root, "visual", 'train_imgs/')
        self.text_feature_path = os.path.join(self.data_root, "text_token", 'train_token/')
        self.stat_path = f"{self.data_root}/stat.txt"
        
        # 根据模态读取对应的CSV文件
        csv_file = f'./data/{args.dataset}/remix_{modality}.csv'
        
        # 读取类别信息
        with open(self.stat_path, "r") as f1:
            classes = [line.strip() for line in f1.readlines()]
        
        self.classes = sorted(classes)
        self.data2class = {}
        self.av_files = []
        
        # 读取模态特定的数据
        with open(csv_file, "r") as f2:
            csv_reader = csv.reader(f2)
            for row in csv_reader:
                if len(row) < 2:
                    continue
                file_name = row[0]
                label_name = self.classes[int(row[1])]
                
                # 验证文件存在
                token_path = os.path.join(self.text_feature_path, file_name + '_token.npy')
                if self.dataset == "MVSA" or self.dataset == "Food101":
                    visual_path = os.path.join(self.visual_feature_path, file_name + ".jpg")
                else:
                    visual_path = os.path.join(self.visual_feature_path, file_name)
                
                if os.path.exists(token_path) and os.path.exists(visual_path):
                    self.av_files.append(file_name)
                    self.data2class[file_name] = label_name
                else:
                    if os.path.exists(token_path):
                        print(f"visual_path is {visual_path}")
                    else:
                        print(f"token_path is {token_path}")

        # 图像变换
        self.preprocess_train = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=True,
            auto_augment=None,
            interpolation="bicubic",
            re_prob=0,
            re_mode=0,
            re_count="const",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        
        # print(f'{modality} data load over')
        # print(f'# of files = {len(self.av_files)}')
        # print(f'# of classes = {len(self.classes)}')
    
    def __len__(self):
        return len(self.av_files)
    
    def get_image(self, filename):
        img = Image.open(filename).convert('RGB')
        image_tensor = self.preprocess_train(img)
        return image_tensor
    
    def __getitem__(self, idx):
        av_file = self.av_files[idx]
        
        # Text
        token_path = os.path.join(self.text_feature_path, av_file + '_token.npy')
        pm_path = os.path.join(self.text_feature_path, av_file + '_pm.npy')
        tokenizer = torch.tensor(np.load(token_path))
        padding_mask = torch.tensor(np.load(pm_path))
        
        # Visual
        if self.dataset == "MVSA" or self.dataset == "Food101":
            image = self.get_image(os.path.join(self.visual_feature_path, av_file + ".jpg"))
        else:
            visual_path = os.path.join(self.visual_feature_path, av_file)
            allimages = os.listdir(visual_path)
            file_num = len(allimages)
            image = self.get_image(os.path.join(visual_path, allimages[int(file_num / 2)]))
        
        label = self.classes.index(self.data2class[av_file])
        
        return tokenizer, padding_mask, image, label, self.modality, av_file


class TVADataset_Remix(Dataset):
    """TVA数据集的Remix版本，支持文本、视觉、音频三种模态特定的训练"""
    
    def __init__(self, args, modality='text', pick_num=3):
        """
        Args:
            args: 参数配置
            modality: 模态名称，如 'text', 'audio', 'visual'
            pick_num: 视觉帧采样数量
        """
        self.args = args
        self.mode = 'train'
        self.pick_num = pick_num
        self.modality = modality
        
        # 分隔符映射
        self.sep_map = {
            "IEMOCAP3": ",",
            "URFUNNY": ",",
        }
        
        # 初始化路径
        self._init_paths()
        
        # 初始化类别
        self._init_classes()
        
        # 初始化数据变换
        self._init_transforms()
        
        # 初始化模态特定数据
        self._init_remix_data()
    
    def _init_paths(self):
        """初始化数据路径"""
        if self.args.dataset == "IEMOCAP3":
            self.data_root = "/root/autodl-tmp/AME/data/dataset/IEMOCAP"
            self.audio_feature_path = os.path.join(self.data_root, "IEMOCAP_full_release", "fbank")
            self.visual_feature_path = os.path.join(self.data_root, "IEMOCAP_full_release", "IMAGE_KEPT_2_PER_SEC")
            self.text_feature_path = os.path.join(self.data_root, "IEMOCAP_full_release", "text_token")
            self.stat_path = os.path.join(self.data_root, "iemocap_stat.txt")
        elif self.args.dataset == "URFUNNY":
            self.data_root = "/path/to/URFUNNY"
            self.audio_feature_path = os.path.join(self.data_root, "audio_features")
            self.visual_feature_path = os.path.join(self.data_root, "visual_features")
            self.text_feature_path = os.path.join(self.data_root, "text_features")
            self.stat_path = os.path.join(self.data_root, "stat.txt")
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")
    
    def _init_classes(self):
        """初始化类别信息"""
        with open(self.stat_path, "r") as f:
            self.classes = sorted([line.strip() for line in f])
        # print(f"Loaded {len(self.classes)} classes: {self.classes}")
    
    def _init_transforms(self):
        """初始化图像变换（Remix始终使用训练模式）"""
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _init_remix_data(self):
        """从CSV文件初始化模态特定数据"""
        self.data = []
        self.data2class = {}
        self.audio_sid_path_map = {}
        self.visual_sid_path_map = {}
        self.text_token_path_map = {}
        self.text_pm_path_map = {}
        self.sid_all_imgs_map = {}
        
        # 根据模态读取对应的CSV文件
        csv_file = f'./data/{self.args.dataset}/remix_{self.modality}.csv'
        with open(csv_file, "r", encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if len(row) < 2:
                    continue
                
                sid = row[0]
                label = self.classes[int(row[1])]
                # 构建文件路径
                audio_path = os.path.join(self.audio_feature_path, f"{sid}.npy")
                visual_path = os.path.join(self.visual_feature_path, sid)
                token_path = os.path.join(self.text_feature_path, f"{sid}_token.pt")
                pm_path = os.path.join(self.text_feature_path, f"{sid}_pm.pt")
                # if idemts == 0:
                #     print(f"audio_path is {audio_path}")
                #     print(f"audio_path is {visual_path}")
                #     print(f"token_path is {token_path}")
                #     print(f"pm_path is {pm_path}")
                #     idemts+=1
                
                # 检查文件存在性
                if not all([os.path.exists(audio_path), os.path.exists(visual_path),
                           os.path.exists(token_path), os.path.exists(pm_path)]):
                    continue
                
                # 检查视觉目录中的图像
                try:
                    image_files = [f for f in os.listdir(visual_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    if not image_files:
                        continue
                    image_files.sort()
                except:
                    # print(f"not image")
                    continue
                
                # 验证标签
                if label not in self.classes:
                    # print(f"not label")
                    continue
                
                # 添加到数据集
                self.data.append(sid)
                self.data2class[sid] = label
                self.audio_sid_path_map[sid] = audio_path
                self.visual_sid_path_map[sid] = visual_path
                self.text_token_path_map[sid] = token_path
                self.text_pm_path_map[sid] = pm_path
                self.sid_all_imgs_map[sid] = image_files
        
        # print(f'{self.modality} data load over')
        # print(f'# of files = {len(self.data)}')
    
    def _load_visual_features(self, visual_path, allimages):
        """加载视觉特征，支持多帧采样"""
        file_num = len(allimages)
        
        if file_num >= self.pick_num:
            seg = file_num / self.pick_num
            indices = [min(int(seg * i), file_num - 1) for i in range(self.pick_num)]
        else:
            indices = [i % file_num for i in range(self.pick_num)]
        
        image_arr = []
        for idx in indices:
            img_path = os.path.join(visual_path, allimages[idx])
            try:
                with Image.open(img_path) as img:
                    image = self.transform(img.convert('RGB')).unsqueeze(1).float()
                image_arr.append(image)
            except:
                dummy_img = torch.zeros(3, 1, 224, 224)
                image_arr.append(dummy_img)
        
        return torch.cat(image_arr, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sid = self.data[idx]
        
        try:
            # 加载文本特征
            tokenizer = torch.load(self.text_token_path_map[sid], map_location='cpu')
            padding_mask = torch.load(self.text_pm_path_map[sid], map_location='cpu')
            
            if tokenizer.dim() > 1:
                tokenizer = tokenizer.squeeze(0)
            if padding_mask.dim() > 1:
                padding_mask = padding_mask.squeeze(0)
            
            # 加载音频特征
            audio_feature = torch.from_numpy(np.load(self.audio_sid_path_map[sid]).astype(np.float32))
            
            # 加载视觉特征
            visual_path = self.visual_sid_path_map[sid]
            allimages = self.sid_all_imgs_map[sid]
            image_n = self._load_visual_features(visual_path, allimages)
            
            # 获取标签
            label = self.classes.index(self.data2class[sid])
            
            return tokenizer, padding_mask, image_n, audio_feature, label, self.modality, sid
            
        except Exception as e:
            print(f"Error loading sample {sid}: {e}")
            dummy_token = torch.zeros(128, dtype=torch.long)
            dummy_mask = torch.zeros(128, dtype=torch.float)
            dummy_image = torch.zeros(3, self.pick_num, 224, 224)
            dummy_audio = torch.zeros(1024, 128)
            dummy_label = 0
            return dummy_token, dummy_mask, dummy_image, dummy_audio, dummy_label, self.modality, sid



class CramedDataset_sample_level(Dataset):
    """CREMA-D数据集的样本级重采样版本"""
    
    def __init__(self, args=None, mode='train', contribution=None):
        """
        Args:
            args: 参数配置
            mode: 训练或测试模式
            contribution: 字典，每个样本的贡献度 {idx: (contrib_a, contrib_v)}
        """
        self.args = args
        self.data = []
        self.drop = []  # 0=none, 1=audio, 2=visual
        self.mode = mode
        classes = []
        data2class = {}
        
        # 数据集路径配置
        self.data_root = "/root/autodl-tmp/AME/data/dataset/CREMA-D/NewData"
        self.visual_path = f'{self.data_root}/Image-01-FPS/'
        self.audio_path = f'{self.data_root}/audio/'
        self.stat_path = f'{self.data_root}/stat.csv'
        self.train_csv = f'{self.data_root}/train.csv'
        self.test_csv = f'{self.data_root}/test.csv'
        
        # 根据模式选择CSV文件
        csv_file = self.train_csv if mode == 'train' else self.test_csv
        
        # 读取类别信息
        with open(self.stat_path, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])
        
        # 加载原始数据
        with open(csv_file, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                if len(item) < 2:
                    continue
                file_name = item[0]
                label_name = item[1]
                
                # 验证文件存在
                audio_path = os.path.join(self.audio_path, file_name + '.npy')
                visual_path = self.visual_path + file_name

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.data.append(file_name)
                    data2class[file_name] = label_name
                    self.drop.append(0)

        self.classes = sorted(classes)
        self.data2class = data2class
        self.fps = 2
        
        print('data load finish')
        length = len(self.data)
        
        # 样本级重采样逻辑（基于每个样本的贡献度）
        if contribution is not None:
            added_count = 0
            contrib_stats = {'audio_high': 0, 'audio_mid': 0, 'audio_low': 0, 
                           'visual_high': 0, 'visual_mid': 0, 'visual_low': 0}
            
            for i in range(length):
                sid = self.data[i]
                contrib_a, contrib_v = contribution.get(sid, (0.0, 0.0))
                
                # 调试信息：打印前几个样本的贡献度
                if i < 3:
                    print(f"Sample {i} (sid={sid}): contrib_a={contrib_a:.4f}, contrib_v={contrib_v:.4f}")
                
                # 根据音频贡献度复制样本并drop视觉
                # 当音频贡献度高时，创建更多audio-only样本（drop visual）
                if 0.4 < contrib_a < 1:  # 音频贡献度高 (复制1次)
                    for _ in range(1):
                        self.data.append(self.data[i])
                        self.drop.append(1)  # drop visual，保留audio
                        added_count += 1
                    contrib_stats['audio_high'] += 1
                elif -0.1 < contrib_a < 0.4:  # 音频贡献度中等 (复制2次)
                    for _ in range(2):
                        self.data.append(self.data[i])
                        self.drop.append(1)
                        added_count += 1
                    contrib_stats['audio_mid'] += 1
                elif contrib_a < -0.1:  # 音频贡献度低 (复制3次)
                    for _ in range(3):
                        self.data.append(self.data[i])
                        self.drop.append(1)
                        added_count += 1
                    contrib_stats['audio_low'] += 1
                
                # 根据视觉贡献度复制样本并drop音频
                # 当视觉贡献度高时，创建更多visual-only样本（drop audio）
                if 0.4 < contrib_v < 1:  # 视觉贡献度高 (复制1次)
                    for _ in range(1):
                        self.data.append(self.data[i])
                        self.drop.append(2)  # drop audio，保留visual
                        added_count += 1
                    contrib_stats['visual_high'] += 1
                elif -0.1 < contrib_v < 0.4:  # 视觉贡献度中等 (复制2次)
                    for _ in range(2):
                        self.data.append(self.data[i])
                        self.drop.append(2)
                        added_count += 1
                    contrib_stats['visual_mid'] += 1
                elif contrib_v < -0.1:  # 视觉贡献度低 (复制3次)
                    for _ in range(3):
                        self.data.append(self.data[i])
                        self.drop.append(2)
                        added_count += 1
                    contrib_stats['visual_low'] += 1
            
            # print(f'data resample finish:')
            # print(f'  Original samples: {length}')
            # print(f'  Added samples: {added_count}')
            # print(f'  Total samples: {len(self.data)}')
            # print(f'  Contribution stats: Audio(high={contrib_stats["audio_high"]}, mid={contrib_stats["audio_mid"]}, low={contrib_stats["audio_low"]}), '
            #       f'Visual(high={contrib_stats["visual_high"]}, mid={contrib_stats["visual_mid"]}, low={contrib_stats["visual_low"]})')
        
        print(f'# of files = {len(self.data)}')
        print(f'# of classes = {len(self.classes)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            tuple: (spectrogram, images, label, sid, drop)
        """
        datum = self.data[idx]

        # 音频处理
        spectrogram = np.load(os.path.join(self.audio_path, datum + '.npy'))
        spectrogram = np.expand_dims(spectrogram, axis=0)

        # 视觉处理
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        folder_path = os.path.join(self.visual_path, datum)
        image_samples = os.listdir(folder_path)
        image_samples.sort()

        if len(image_samples) < self.fps:
            select_index = np.random.choice(len(image_samples), size=self.fps, replace=True)
        else:
            select_index = np.random.choice(len(image_samples), size=self.fps, replace=False)
        select_index.sort()

        images = torch.zeros((self.fps, 3, 224, 224), dtype=torch.float32)
        for i in range(self.fps):
            img_path = os.path.join(folder_path, image_samples[select_index[i]])
            img = Image.open(img_path).convert('RGB')
            images[i] = transform(img)

        images = torch.permute(images, (1, 0, 2, 3))
        label = self.classes.index(self.data2class[datum])
        drop = self.drop[idx]
        
        return spectrogram, images, label, datum, drop


class AVEDataset_sample_level(Dataset):
    """AVE数据集的样本级重采样版本"""
    
    def __init__(self, args, mode='train', contribution=None):
        """
        Args:
            args: 参数配置
            mode: 训练或测试模式
            contribution: 字典，每个样本的贡献度 {idx: (contrib_a, contrib_v)}
        """
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.drop = []
        self.mode = mode
        classes = []
        
        self.data_root = '/root/autodl-tmp/AME/data/dataset/AVE/AVE_Dataset'
        self.visual_feature_path = os.path.join(self.data_root, 'Image-01-FPS-SE')
        self.audio_feature_path = os.path.join(self.data_root, 'Audio-1004-SE')
        
        if mode == 'train':
            csv_file = os.path.join(self.data_root, 'trainSet.txt')
        else:
            csv_file = os.path.join(self.data_root, 'testSet.txt')
        
        # 读取类别信息
        with open(csv_file, 'r') as f1:
            files = f1.readlines()
            for item in files:
                item = item.split('&')
                if item[0] not in classes:
                    classes.append(item[0])
        class_dict = {}
        for i, c in enumerate(classes):
            class_dict[c] = i
        
        # 加载原始数据
        with open(csv_file, 'r') as f2:
            for row in f2:
                item = row.strip().split('&')
                if len(item) < 3:
                    continue
                label_name = item[0]
                audio_name = item[1]
                
                # 验证文件存在
                audio_path = os.path.join(self.audio_feature_path, audio_name + '.pkl')
                visual_path = os.path.join(self.visual_feature_path, audio_name)
                
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[label_name])
                    self.drop.append(0)
        
        print('data load finish')
        length = len(self.image)
        
        # 样本级重采样逻辑
        if contribution is not None:
            for i in range(length):
                contrib_a, contrib_v = contribution.get(i, (0.0, 0.0))
                
                # 根据音频贡献度复制样本
                if 0.4 < contrib_a < 1:
                    for _ in range(1):
                        self.image.append(self.image[i])
                        self.audio.append(self.audio[i])
                        self.label.append(self.label[i])
                        self.drop.append(2)  # drop visual
                elif -0.1 < contrib_a < 0.4:
                    for _ in range(2):
                        self.image.append(self.image[i])
                        self.audio.append(self.audio[i])
                        self.label.append(self.label[i])
                        self.drop.append(2)
                elif contrib_a < -0.1:
                    for _ in range(3):
                        self.image.append(self.image[i])
                        self.audio.append(self.audio[i])
                        self.label.append(self.label[i])
                        self.drop.append(2)
                
                # 根据视觉贡献度复制样本
                if 0.4 < contrib_v < 1:
                    for _ in range(1):
                        self.image.append(self.image[i])
                        self.audio.append(self.audio[i])
                        self.label.append(self.label[i])
                        self.drop.append(1)  # drop audio
                elif -0.1 < contrib_v < 0.4:
                    for _ in range(2):
                        self.image.append(self.image[i])
                        self.audio.append(self.audio[i])
                        self.label.append(self.label[i])
                        self.drop.append(1)
                elif contrib_v < -0.1:
                    for _ in range(3):
                        self.image.append(self.image[i])
                        self.audio.append(self.audio[i])
                        self.label.append(self.label[i])
                        self.drop.append(1)
            
            print('data resample finish')
        
        print(f'# of files = {len(self.image)}')
        print(f'# of classes = {len(classes)}')
    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, idx):
        # 音频处理
        spectrogram = pickle.load(open(self.audio[idx], 'rb'))
        
        # 根据模式选择transform
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # 视觉处理
        image_samples = os.listdir(self.image[idx])
        image_samples.sort()
        
        if len(image_samples) < self.args.fps:
            select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=True)
        else:
            select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=False)
        select_index.sort()
        
        images = torch.zeros((self.args.fps, 3, 224, 224), dtype=torch.float32)
        for i in range(self.args.fps):
            img = Image.open(os.path.join(self.image[idx], image_samples[select_index[i]])).convert('RGB')
            img = transform(img)
            images[i] = img
        
        images = torch.permute(images, (1, 0, 2, 3))
        label = self.label[idx]
        sid = self.audio[idx].split('/')[-1].split('.')[0]
        drop = self.drop[idx]
        
        return spectrogram, images, label, sid, drop


class M3AEDataset_sample_level(Dataset):
    """M3AE数据集（Food101/MVSA）的样本级重采样版本"""
    
    def __init__(self, args, mode='train', contribution=None):
        """
        Args:
            args: 参数配置
            mode: 训练或测试模式
            contribution: 字典，每个样本的贡献度 {idx: (contrib_t, contrib_v)}
        """
        self.args = args
        self.mode = mode
        self.dataset = args.dataset
        self.drop = []
        
        # 路径配置
        if args.dataset == "Food101":
            self.data_root = '/root/autodl-tmp/AME/data/dataset/Food101/Food101'
        elif args.dataset == "MVSA":
            self.data_root = '/root/autodl-tmp/AME/data/dataset/MVSA/MVSA_Single'
        
        if mode == 'train':
            self.visual_feature_path = os.path.join(self.data_root, "visual", 'train_imgs/')
            self.text_feature_path = os.path.join(self.data_root, "text_token", 'train_token/')
            csv_file = os.path.join(self.data_root, 'train.csv')
        else:
            self.visual_feature_path = os.path.join(self.data_root, "visual", 'test_imgs/')
            self.text_feature_path = os.path.join(self.data_root, "text_token", 'test_token/')
            csv_file = os.path.join(self.data_root, 'test.csv')
        
        self.stat_path = f"{self.data_root}/stat.txt"
        
        # 读取类别信息
        with open(self.stat_path, "r") as f1:
            classes = [line.strip() for line in f1.readlines()]
        
        self.classes = sorted(classes)
        self.data2class = {}
        self.av_files = []
        
        # 加载原始数据
        with open(csv_file, "r") as f2:
            csv_reader = csv.reader(f2)
            for row in csv_reader:
                if len(row) < 2:
                    continue
                file_name = row[0]
                label_name = row[1]
                
                # 验证文件存在
                token_path = os.path.join(self.text_feature_path, file_name + '_token.npy')
                if self.dataset == "MVSA" or self.dataset == "Food101":
                    visual_path = os.path.join(self.visual_feature_path, file_name + ".jpg")
                else:
                    visual_path = os.path.join(self.visual_feature_path, file_name)
                
                if os.path.exists(token_path) and os.path.exists(visual_path):
                    self.av_files.append(file_name)
                    self.data2class[file_name] = label_name
                    self.drop.append(0)
        
        print('data load finish')
        length = len(self.av_files)
        
        # 样本级重采样逻辑（文本和视觉）
        if contribution is not None:
            for i in range(length):
                contrib_t, contrib_v = contribution.get(i, (0.0, 0.0))
                
                # 根据文本贡献度复制样本
                if 0.4 < contrib_t < 1:
                    for _ in range(1):
                        self.av_files.append(self.av_files[i])
                        self.drop.append(2)  # drop visual
                elif -0.1 < contrib_t < 0.4:
                    for _ in range(2):
                        self.av_files.append(self.av_files[i])
                        self.drop.append(2)
                elif contrib_t < -0.1:
                    for _ in range(3):
                        self.av_files.append(self.av_files[i])
                        self.drop.append(2)
                
                # 根据视觉贡献度复制样本
                if 0.4 < contrib_v < 1:
                    for _ in range(1):
                        self.av_files.append(self.av_files[i])
                        self.drop.append(1)  # drop text
                elif -0.1 < contrib_v < 0.4:
                    for _ in range(2):
                        self.av_files.append(self.av_files[i])
                        self.drop.append(1)
                elif contrib_v < -0.1:
                    for _ in range(3):
                        self.av_files.append(self.av_files[i])
                        self.drop.append(1)
            
            print('data resample finish')
        
        print(f'# of files = {len(self.av_files)}')
        print(f'# of classes = {len(self.classes)}')

        # 图像变换
        self.preprocess_train = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=True,
            auto_augment=None,
            interpolation="bicubic",
            re_prob=0,
            re_mode=0,
            re_count="const",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        
        # print(f'{modality} data load over')
        # print(f'# of files = {len(self.av_files)}')
        # print(f'# of classes = {len(self.classes)}')
    
    def __len__(self):
        return len(self.av_files)
    
    def get_image(self, filename):
        img = Image.open(filename).convert('RGB')
        image_tensor = self.preprocess_train(img)
        return image_tensor
    
    def __getitem__(self, idx):
        av_file = self.av_files[idx]
        
        # Text
        token_path = os.path.join(self.text_feature_path, av_file + '_token.npy')
        pm_path = os.path.join(self.text_feature_path, av_file + '_pm.npy')
        tokenizer = torch.tensor(np.load(token_path))
        padding_mask = torch.tensor(np.load(pm_path))
        
        # Visual
        if self.dataset == "MVSA" or self.dataset == "Food101":
            image = self.get_image(os.path.join(self.visual_feature_path, av_file + ".jpg"))
        else:
            visual_path = os.path.join(self.visual_feature_path, av_file)
            allimages = os.listdir(visual_path)
            file_num = len(allimages)
            image = self.get_image(os.path.join(visual_path, allimages[int(file_num / 2)]))
        
        label = self.classes.index(self.data2class[av_file])
        drop = self.drop[idx]
        
        return tokenizer, padding_mask, image, label, av_file, drop


class TVADataset_sample_level(Dataset):
    """TVA数据集的样本级重采样版本"""
    
    def __init__(self, args, mode='train', pick_num=3, contribution=None):
        """
        Args:
            args: 参数配置
            mode: 训练或测试模式
            pick_num: 视觉帧采样数量
            contribution: 字典，每个样本的贡献度 {idx: (contrib_t, contrib_v, contrib_a)}
        """
        self.args = args
        self.mode = mode
        self.pick_num = pick_num
        self.drop = []
        
        # 分隔符映射
        self.sep_map = {
            "IEMOCAP3": ",",
            "URFUNNY": ",",
        }
        
        # 初始化路径
        self._init_paths()
        
        # 初始化类别
        self._init_classes()
        
        # 初始化数据变换
        self._init_transforms()
        
        # 初始化数据（含重采样逻辑）
        self.contribution = contribution
        self._init_sample_level_data()
    
    def _init_paths(self):
        """初始化数据路径"""
        if self.args.dataset == "IEMOCAP3":
            self.data_root = "/root/autodl-tmp/AME/data/dataset/IEMOCAP"
            self.audio_feature_path = os.path.join(self.data_root, "IEMOCAP_full_release", "fbank")
            self.visual_feature_path = os.path.join(self.data_root, "IEMOCAP_full_release", "IMAGE_KEPT_2_PER_SEC")
            self.text_feature_path = os.path.join(self.data_root, "IEMOCAP_full_release", "text_token")
            self.stat_path = os.path.join(self.data_root, "iemocap_stat.txt")
        elif self.args.dataset == "URFUNNY":
            self.data_root = "/path/to/URFUNNY"
            self.audio_feature_path = os.path.join(self.data_root, "audio_features")
            self.visual_feature_path = os.path.join(self.data_root, "visual_features")
            self.text_feature_path = os.path.join(self.data_root, "text_features")
            self.stat_path = os.path.join(self.data_root, "stat.txt")
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")
    
    def _init_classes(self):
        """初始化类别信息"""
        with open(self.stat_path, "r") as f:
            self.classes = sorted([line.strip() for line in f])
        # print(f"Loaded {len(self.classes)} classes: {self.classes}")
    
    def _init_transforms(self):
        """初始化图像变换（Remix始终使用训练模式）"""
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _init_sample_level_data(self):
        """初始化样本级数据并执行重采样"""
        self.data = []
        self.data2class = {}
        self.audio_sid_path_map = {}
        self.visual_sid_path_map = {}
        self.text_token_path_map = {}
        self.text_pm_path_map = {}
        self.sid_all_imgs_map = {}
        
        # 根据模式选择CSV文件
        if self.mode == 'train':
            csv_file = f'./data/{self.args.dataset}/train.csv'
        else:
            csv_file = f'./data/{self.args.dataset}/test.csv'
        
        with open(csv_file, "r", encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                if len(row) < 2:
                    continue
                
                sid = row[0]
                label = self.classes[int(row[1])]
                # 构建文件路径
                audio_path = os.path.join(self.audio_feature_path, f"{sid}.npy")
                visual_path = os.path.join(self.visual_feature_path, sid)
                token_path = os.path.join(self.text_feature_path, f"{sid}_token.pt")
                pm_path = os.path.join(self.text_feature_path, f"{sid}_pm.pt")
                # if idemts == 0:
                #     print(f"audio_path is {audio_path}")
                #     print(f"audio_path is {visual_path}")
                #     print(f"token_path is {token_path}")
                #     print(f"pm_path is {pm_path}")
                #     idemts+=1
                
                # 检查文件存在性
                if not all([os.path.exists(audio_path), os.path.exists(visual_path),
                           os.path.exists(token_path), os.path.exists(pm_path)]):
                    continue
                
                # 检查视觉目录中的图像
                try:
                    image_files = [f for f in os.listdir(visual_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    if not image_files:
                        continue
                    image_files.sort()
                except:
                    # print(f"not image")
                    continue
                
                # 验证标签
                if label not in self.classes:
                    # print(f"not label")
                    continue
                
                # 添加到数据集
                self.data.append(sid)
                self.data2class[sid] = label
                self.audio_sid_path_map[sid] = audio_path
                self.visual_sid_path_map[sid] = visual_path
                self.text_token_path_map[sid] = token_path
                self.text_pm_path_map[sid] = pm_path
                self.sid_all_imgs_map[sid] = image_files
                self.drop.append(0)
        
        print('data load finish')
        length = len(self.data)
        
        # 样本级重采样逻辑（文本、视觉、音频）
        if self.contribution is not None:
            for i in range(length):
                contrib_t, contrib_v, contrib_a = self.contribution.get(i, (0.0, 0.0, 0.0))
                
                # 根据文本贡献度复制样本
                if 0.4 < contrib_t < 1:
                    self._append_sample(i, 1, 3)  # drop视觉和音频
                elif -0.1 < contrib_t < 0.4:
                    self._append_sample(i, 2, 3)
                elif contrib_t < -0.1:
                    self._append_sample(i, 3, 3)
                
                # 根据视觉贡献度复制样本
                if 0.4 < contrib_v < 1:
                    self._append_sample(i, 1, 2)  # drop文本和音频
                elif -0.1 < contrib_v < 0.4:
                    self._append_sample(i, 2, 2)
                elif contrib_v < -0.1:
                    self._append_sample(i, 3, 2)
                
                # 根据音频贡献度复制样本
                if 0.4 < contrib_a < 1:
                    self._append_sample(i, 1, 1)  # drop文本和视觉
                elif -0.1 < contrib_a < 0.4:
                    self._append_sample(i, 2, 1)
                elif contrib_a < -0.1:
                    self._append_sample(i, 3, 1)
            
            print('data resample finish')
        
        print(f'# of files = {len(self.data)}')
    
    def _append_sample(self, idx, times, drop_val):
        """辅助函数：复制样本"""
        for _ in range(times):
            sid = self.data[idx]
            self.data.append(sid)
            self.drop.append(drop_val)
    
    def _load_visual_features(self, visual_path, allimages):
        """加载视觉特征，支持多帧采样"""
        file_num = len(allimages)
        
        if file_num >= self.pick_num:
            seg = file_num / self.pick_num
            indices = [min(int(seg * i), file_num - 1) for i in range(self.pick_num)]
        else:
            indices = [i % file_num for i in range(self.pick_num)]
        
        image_arr = []
        for idx in indices:
            img_path = os.path.join(visual_path, allimages[idx])
            try:
                with Image.open(img_path) as img:
                    image = self.transform(img.convert('RGB')).unsqueeze(1).float()
                image_arr.append(image)
            except:
                dummy_img = torch.zeros(3, 1, 224, 224)
                image_arr.append(dummy_img)
        
        return torch.cat(image_arr, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sid = self.data[idx]
        
        try:
            # 加载文本特征
            tokenizer = torch.load(self.text_token_path_map[sid], map_location='cpu')
            padding_mask = torch.load(self.text_pm_path_map[sid], map_location='cpu')
            
            if tokenizer.dim() > 1:
                tokenizer = tokenizer.squeeze(0)
            if padding_mask.dim() > 1:
                padding_mask = padding_mask.squeeze(0)
            
            # 加载音频特征
            audio_feature = torch.from_numpy(np.load(self.audio_sid_path_map[sid]).astype(np.float32))
            
            # 加载视觉特征
            visual_path = self.visual_sid_path_map[sid]
            allimages = self.sid_all_imgs_map[sid]
            image_n = self._load_visual_features(visual_path, allimages)
            
            # 获取标签和drop
            label = self.classes.index(self.data2class[sid])
            drop = self.drop[idx]
            
            return tokenizer, padding_mask, image_n, audio_feature, label, sid, drop
            
        except Exception as e:
            print(f"Error loading sample {sid}: {e}")
            dummy_token = torch.zeros(128, dtype=torch.long)
            dummy_mask = torch.zeros(128, dtype=torch.float)
            dummy_image = torch.zeros(3, self.pick_num, 224, 224)
            dummy_audio = torch.zeros(1024, 128)
            dummy_label = 0
            dummy_drop = 0
            return dummy_token, dummy_mask, dummy_image, dummy_audio, dummy_label, sid, dummy_drop
        