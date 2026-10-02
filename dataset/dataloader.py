
# © 2025 Xiaoyu Ma. 
# Code of Improving Multimodal Learning Balance and Sufficiency through Data Remixing.
# This code is adapted from OGM-GE, available at:
# https://github.com/GeWu-Lab/OGM-GE_CVPR2022
# All rights reserved.

import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import random
import copy
import csv

class AV_CD_Dataset(Dataset):
    def __init__(self, mode='train'):
        classes = []
        self.data = []
        data2class = {}

        self.mode=mode
        self.data_root = "/root/autodl-tmp/AME/data/dataset/CREMA-D/NewData"
        self.visual_path = f'{self.data_root}/visual/'
        self.audio_path = f'{self.data_root}/audio/'
        self.stat_path = f'{self.data_root}/stat.csv'
        self.train_txt = f'{self.data_root}/train.csv'
        self.test_txt = f'{self.data_root}/test.csv'

        if mode == 'train' or mode == 'val':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(self.stat_path, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])
        
        with open(csv_file) as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                # print(f"audio path is {self.audio_path + item[0] + '.npy'}")
                # break
                if item[1] in classes and os.path.exists(self.audio_path + item[0] + '.npy') and os.path.exists(
                                self.visual_path + item[0]):
                    self.data.append(item[0])
                    data2class[item[0]] = item[1]
        
        self.classes = sorted(classes)
        self.data2class = data2class
        self._init_atransform()

        print('data load over')
        print('# of files = %d ' % len(self.data))
        print('# of classes = %d' % len(self.classes))

    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

  
    def __getitem__(self, idx):
        # datum: file name without .xxx
        datum = self.data[idx]

        # Audio
        spectrogram = np.load(os.path.join(self.audio_path, datum + '.npy'))
        spectrogram = np.expand_dims(spectrogram, axis=0)

        # # Visual
        # if self.mode == 'train':
        #     transf = transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ])

        # else:
        #     transf = transforms.Compose([
        #         transforms.Resize(size=(224, 224)),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ])

        # folder_path = os.path.join(self.visual_path, datum)
        # file_num = len(os.listdir(folder_path))
        # pick_num = 2
        # seg = int(file_num/pick_num)

        # for i in range(pick_num):
        #     if self.mode == 'train':
        #         # Ensure selected index <= the largest index
        #         start_index = i * seg
        #         end_index = min((i + 1) * seg - 1, file_num - 1)  
        #         index = random.randint(start_index, end_index)
        #     else:
        #         index = min(i * seg + int(seg / 2), file_num - 1)
                
        #     path = os.path.join(folder_path, 'frame_0000' + str(index+1) + '.jpg')
        #     image_arr = transf(Image.open(path).convert('RGB')).unsqueeze(1).float()

        #     if i == 0:
        #         image_n = copy.copy(image_arr)
        #     else:
        #         image_n = torch.cat((image_n, image_arr), 1)
        # -------------------------- 视觉处理（核心修复：正确使用随机索引）--------------------------
        # 定义图像变换（训练模式含数据增强，测试模式仅固定缩放）
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),          # 训练增强：随机裁剪至224×224
                transforms.RandomHorizontalFlip(),           # 训练增强：50%概率水平翻转
                transforms.ToTensor(),                      # 转为张量（形状：[3, 224, 224]）
                transforms.Normalize([0.485, 0.456, 0.406],  # 用ImageNet预训练均值/标准差归一化
                                     [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),              # 测试模式：固定缩放至224×224
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        folder_path = os.path.join(self.visual_path, datum)
        # 1. 获取当前样本的所有帧图像文件名，并按字母/数字排序（确保帧的时间顺序）
        image_samples = os.listdir(folder_path)
        image_samples.sort()  # 关键：避免os.listdir返回顺序混乱导致帧顺序错误
        fps = 2
        # 2. 随机选择self.args.fps帧（不重复），并按时间顺序排序
        # 若帧数量不足fps，允许重复采样（避免报错）
        if len(image_samples) < fps:
            select_index = np.random.choice(len(image_samples), size=fps, replace=True)
        else:
            select_index = np.random.choice(len(image_samples), size=fps, replace=False)
        select_index.sort()  # 排序索引：确保采样的帧按时间顺序排列，符合时序逻辑

        # 3. 加载并处理选中的帧（核心修复：使用select_index[i]而非i作为索引）
        images = torch.zeros((fps, 3, 224, 224), dtype=torch.float32)
        for i in range(fps):
            # 修复点：用随机生成的select_index[i]选择图像，而非固定取前fps张
            img_path = os.path.join(folder_path, image_samples[select_index[i]])
            # 加载图像并转为RGB通道（避免灰度图导致的通道数异常）
            img = Image.open(img_path).convert('RGB')
            # 应用图像变换
            images[i] = transform(img)

        # 调整维度顺序：从[fps, 3, 224, 224]转为[3, fps, 224, 224]（通道优先，适配模型输入）
        images = torch.permute(images, (1, 0, 2, 3))
        return spectrogram, images, self.classes.index(self.data2class[datum]), self.data2class[datum], datum


class AV_CD_Dataset_Remix(Dataset):
    def __init__(self, modality=None):

        classes = []
        self.data = []
        data2class = {}

        self.modality = modality
        self.data_root = "/root/autodl-tmp/AME/data/dataset/CREMA-D/NewData"
        self.visual_path = f'{self.data_root}/Image-01-FPS/'
        self.audio_path = f'{self.data_root}/audio/'
        self.stat_path = f'{self.data_root}/stat.csv'
        # self.train_txt = f'{self.data_root}/train.csv'
        # self.test_txt = f'{self.data_root}/test.csv'
        self.audio_specific_train_txt = './data/remix_a.csv'
        self.video_specific_train_txt = './data/remix_v.csv'

        if modality == 'audio':
            csv_file = self.audio_specific_train_txt
        else:
            csv_file = self.video_specific_train_txt

        with open(self.stat_path, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])
        
        with open(csv_file) as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                if item[1] in classes and os.path.exists(self.audio_path + item[0] + '.npy') and os.path.exists(
                                self.visual_path + item[0]):
                    self.data.append(item[0])
                    data2class[item[0]] = item[1]

        self.classes = sorted(classes)
        self.data2class = data2class
        self._init_atransform()
        print(f'{modality} data load over')
        print('# of files = %d ' % len(self.data))
        print('# of classes = %d' % len(self.classes))


    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):

        # Same as normal dataset, modality masking will be adopted in model.forward()
        datum = self.data[idx]

        # Audio
        spectrogram = np.load(os.path.join(self.audio_path, datum + '.npy'))
        spectrogram = np.expand_dims(spectrogram, axis=0)

        # Visual
         # -------------------------- 视觉处理（核心修复：正确使用随机索引）--------------------------
        # 定义图像变换（训练模式含数据增强，测试模式仅固定缩放）
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),          # 训练增强：随机裁剪至224×224
            transforms.RandomHorizontalFlip(),           # 训练增强：50%概率水平翻转
            transforms.ToTensor(),                      # 转为张量（形状：[3, 224, 224]）
            transforms.Normalize([0.485, 0.456, 0.406],  # 用ImageNet预训练均值/标准差归一化
                                 [0.229, 0.224, 0.225])
        ])
       
        folder_path = os.path.join(self.visual_path, datum)
        # 1. 获取当前样本的所有帧图像文件名，并按字母/数字排序（确保帧的时间顺序）
        image_samples = os.listdir(folder_path)
        image_samples.sort()  # 关键：避免os.listdir返回顺序混乱导致帧顺序错误
        fps = 2
        # 2. 随机选择self.args.fps帧（不重复），并按时间顺序排序
        # 若帧数量不足fps，允许重复采样（避免报错）
        if len(image_samples) < fps:
            select_index = np.random.choice(len(image_samples), size=fps, replace=True)
        else:
            select_index = np.random.choice(len(image_samples), size=fps, replace=False)
        select_index.sort()  # 排序索引：确保采样的帧按时间顺序排列，符合时序逻辑

        # 3. 加载并处理选中的帧（核心修复：使用select_index[i]而非i作为索引）
        images = torch.zeros((fps, 3, 224, 224), dtype=torch.float32)
        for i in range(fps):
            # 修复点：用随机生成的select_index[i]选择图像，而非固定取前fps张
            img_path = os.path.join(folder_path, image_samples[select_index[i]])
            # 加载图像并转为RGB通道（避免灰度图导致的通道数异常）
            img = Image.open(img_path).convert('RGB')
            # 应用图像变换
            images[i] = transform(img)

        # 调整维度顺序：从[fps, 3, 224, 224]转为[3, fps, 224, 224]（通道优先，适配模型输入）
        images = torch.permute(images, (1, 0, 2, 3))
        return images, spectrogram, self.classes.index(self.data2class[datum]), self.data2class[datum], datum