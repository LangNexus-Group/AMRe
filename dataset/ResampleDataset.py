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


# ==================== Modality Level Datasets ====================
class CramedDataset_modality_level(Dataset):
    """AV_CD_Dataset的模态级重采样版本"""
    
    def __init__(self, args=None, mode='train', contribution_a=0.5, contribution_v=0.5, alpha=1.0, func='linear'):
        """
        Args:
            args: 参数配置（可选）
            mode: 训练或测试模式
            contribution_a: 音频模态贡献度
            contribution_v: 视觉模态贡献度
            alpha: 重采样强度系数
            func: 差异计算函数 ('linear', 'tanh', 'square')
        """
        self.args = args
        self.data = []
        self.drop = []  # 0=none, 1=audio, 2=visual
        self.mode = mode
        classes = []
        data2class = {}
        
        # 数据集路径配置
        self.data_root = "/root/autodl-tmp/AME/data/dataset/CREMA-D/NewData"
        self.visual_path = f'{self.data_root}/visual/'
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
                audio_path = os.path.join(self.audio_path, item[0] + '.npy')
                visual_path = os.path.join(self.visual_path, item[0])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.data.append(item[0])
                    data2class[item[0]] = item[1]
                    self.drop.append(0)

        self.classes = sorted(classes)
        self.data2class = data2class
        self.fps = 2  # 默认2帧
        
        print('data load finish')
        length = len(self.data)

        # 模态级重采样逻辑
        gap_a = 1.0 - contribution_a
        gap_v = 1.0 - contribution_v

        if func == 'linear':
            difference = (abs(gap_a - gap_v) / 3 * 2) * alpha
        elif func == 'tanh':
            tanh = torch.nn.Tanh()
            difference = tanh(torch.tensor((abs(gap_a - gap_v) / 3 * 2) * alpha)).item()
        elif func == 'square':
            difference = (abs(gap_a - gap_v) / 3 * 2) ** 1.5 * alpha
        else:
            difference = (abs(gap_a - gap_v) / 3 * 2) * alpha
        
        resample_num = int(difference * length)
        sample_choice = np.random.choice(length, resample_num)

        for i in sample_choice:
            self.data.append(self.data[i])
            if gap_a > gap_v:
                self.drop.append(2)  # drop visual
            else:
                self.drop.append(1)  # drop audio

        print('data resample finish')
        print(f'# of files = {len(self.data)}')
        print(f'# of classes = {len(self.classes)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据项
        
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
        sid = datum

        return spectrogram, images, label, sid, 0 ,drop


class KSDataset_modality_level(nn.Module):
    """KineticSound数据集的模态级重采样版本"""
    
    def __init__(self, args, mode='train', contribution_a=0.5, contribution_v=0.5, alpha=1.0, func='linear'):
        super().__init__()
        self.args = args
        self.labels = []
        self.videos = []
        self.audios = []
        self.drop = []
        
        data_path = '/root/autodl-tmp/AME/data/dataset/KSDataset/kinect_sound'
        f = open(f'{data_path}/class.txt')
        data = f.readline()
        class_list = data.split(',')
        for i in range(len(class_list)):
            if " " in class_list[i]:
                class_name = class_list[i].split(" ")
                if class_name[0] == '':
                    class_name = class_name[1:len(class_name)]
                class_name = '_'.join(class_name)
                class_list[i] = class_name
        
        labels_list = range(len(class_list))
        data_dict = zip(class_list, labels_list)
        data_dict = dict(data_dict)

        self.mode = mode
        if self.mode == 'train':
            visual_data_path = os.path.join(data_path, 'visual', 'train_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'train')
        elif self.mode == 'test':
            visual_data_path = os.path.join(data_path, 'visual', 'val_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'test')

        remove_list = []

        for class_name in class_list:
            visual_class_path = os.path.join(visual_data_path, class_name)
            audio_class_path = os.path.join(audio_data_path, class_name)

            video_list = os.listdir(visual_class_path)
            video_list.sort()

            audio_list = os.listdir(audio_class_path)
            audio_list.sort()

            for video in video_list:
                video_path = os.path.join(visual_class_path, video)

                if len(listdir_nohidden(video_path)) < 3:
                    remove_list.append(video)
                    continue

                self.videos.append(video_path)
                self.labels.append(data_dict[class_name])
                self.drop.append(0)

            for audio in audio_list:
                if audio in remove_list:
                    continue
                audio_path = os.path.join(audio_class_path, audio)
                self.audios.append(audio_path)

        print('data load finish')
        length = len(self.labels)

        # 模态级重采样
        gap_a = 1.0 - contribution_a
        gap_v = 1.0 - contribution_v

        if func == 'linear':
            difference = (abs(gap_a - gap_v) / 3 * 2) * alpha
        elif func == 'tanh':
            tanh = torch.nn.Tanh()
            difference = tanh(torch.tensor((abs(gap_a - gap_v) / 3 * 2) * alpha))
        elif func == 'square':
            difference = (abs(gap_a - gap_v) / 3 * 2) ** 1.5 * alpha
        
        resample_num = int(difference * length)
        sample_choice = np.random.choice(length, resample_num)

        for i in sample_choice:
            self.videos.append(self.videos[i])
            self.audios.append(self.audios[i])
            self.labels.append(self.labels[i])
            if gap_a > gap_v:
                self.drop.append(2)
            else:
                self.drop.append(1)

        print('data resample finish')
        print('# of files = %d ' % len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # audio
        sample, rate = librosa.load(self.audios[idx], sr=16000, mono=True)
        while len(sample) / rate < 10.:
            sample = np.tile(sample, 2)

        start_point = random.randint(a=0, b=rate * 5)
        new_sample = sample[start_point:start_point + rate * 5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)
        spectrogram = np.transpose(spectrogram, (1, 0))
        spectrogram = np.transpose(spectrogram, (1, 0))

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = listdir_nohidden(self.videos[idx])
        select_index = np.random.choice(len(image_samples), size=self.args.use_video_frames, replace=False)
        select_index.sort()
        images = torch.zeros((self.args.use_video_frames, 3, 224, 224))
        for i in range(self.args.use_video_frames):
            try:
                img = Image.open(image_samples[select_index[i]]).convert('RGB')
            except Exception as e:
                print(e)
                print(image_samples[select_index[i]])
                continue
            img = transform(img)
            images[i] = img

        images = torch.permute(images, (1, 0, 2, 3))

        label = self.labels[idx]
        drop = self.drop[idx]
        sid = self.audios[idx].split('/')[-1].split('.')[0]

        return spectrogram, images, label, sid, drop


class AVEDataset_modality_level(Dataset):
    """AVE数据集的模态级重采样版本"""
    
    def __init__(self, args, mode='train', contribution_a=0.5, contribution_v=0.5, alpha=1.0, func='linear'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.drop = []
        self.mode = mode
        classes = []
        self.data_root = '/root/autodl-tmp/AME/data/dataset/AVE/AVE_Dataset'

        self.visual_feature_path = self.data_root
        self.audio_feature_path = os.path.join(self.data_root, 'Audio-1004-SE')

        self.train_txt = os.path.join(self.data_root, 'trainSet.txt')
        self.test_txt = os.path.join(self.data_root, 'testSet.txt')
        self.val_txt = os.path.join(self.data_root, 'valSet.txt')

        if mode == 'train':
            txt_file = self.train_txt
        elif mode == 'test':
            txt_file = self.test_txt
        else:
            txt_file = self.val_txt

        with open(self.test_txt, 'r') as f1:
            files = f1.readlines()
            for item in files:
                item = item.split('&')
                if item[0] not in classes:
                    classes.append(item[0])
        class_dict = {}
        for i, c in enumerate(classes):
            class_dict[c] = i

        with open(txt_file, 'r') as f2:
            files = f2.readlines()
            for item in files:
                item = item.split('&')
                audio_path = os.path.join(self.audio_feature_path, item[1] + '.pkl')
                visual_path = os.path.join(self.visual_feature_path, 'Image-01-FPS-SE', item[1])

                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    if os.stat(audio_path).st_size < 200:
                        continue
                    if audio_path not in self.audio:
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[item[0]])
                        self.drop.append(0)

        print('data load finish')
        length = len(self.image)

        # 模态级重采样
        gap_a = 1.0 - contribution_a
        gap_v = 1.0 - contribution_v

        if func == 'linear':
            difference = (abs(gap_a - gap_v) / 3 * 2) * alpha
        elif func == 'tanh':
            tanh = torch.nn.Tanh()
            difference = tanh(torch.tensor((abs(gap_a - gap_v) / 3 * 2) * alpha))
        elif func == 'square':
            difference = (abs(gap_a - gap_v) / 3 * 2) ** 1.5 * alpha
        
        resample_num = int(difference * length)
        sample_choice = np.random.choice(length, resample_num)

        for i in sample_choice:
            self.image.append(self.image[i])
            self.audio.append(self.audio[i])
            self.label.append(self.label[i])
            if gap_a > gap_v:
                self.drop.append(2)
            else:
                self.drop.append(1)

        print('data resample finish')
        print('# of files = %d ' % len(self.image))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        # Audio
        spectrogram = pickle.load(open(self.audio[idx], 'rb'))

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        image_samples.sort()

        if len(image_samples) < self.args.fps:
            select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=True)
        else:
            select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=False)
        select_index.sort()

        images = torch.zeros((self.args.fps, 3, 224, 224), dtype=torch.float32)
        for i in range(self.args.fps):
            img_path = os.path.join(self.image[idx], image_samples[select_index[i]])
            img = Image.open(img_path).convert('RGB')
            images[i] = transform(img)

        images = torch.permute(images, (1, 0, 2, 3))
        label = self.label[idx]
        drop = self.drop[idx]
        sid = self.audio[idx].split('/')[-1].split('.')[0]

        return spectrogram, images, label, sid, drop


class M3AEDataset_modality_level(Dataset):
    """M3AE数据集（Food101/MVSA）的模态级重采样版本"""
    
    def __init__(self, args, mode='train', contribution_t=0.5, contribution_v=0.5, alpha=1.0, func='linear'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode
        self.dataset = args.dataset
        self.drop = []
        
        if args.dataset == "Food101":
            self.data_root = '/root/autodl-tmp/AME/data/dataset/Food101/food101'
        elif args.dataset == "MVSA":
            self.data_root = '/root/autodl-tmp/AME/data/dataset/MVSA/MVSA_Single'
            
        self.visual_feature_path = os.path.join(self.data_root, "visual", '{}_imgs/'.format(mode))
        self.text_feature_path = os.path.join(self.data_root, "text_token", '{}_token/'.format(mode))
        self.stat_path = f"{self.data_root}/stat.txt"
        self.train_txt = f"{self.data_root}/my_train.txt"
        self.test_txt = f"{self.data_root}/my_test.txt"

        with open(self.stat_path, "r") as f1:
            classes = f1.readlines()
        
        classes = [sclass.strip() for sclass in classes]

        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(csv_file, "r") as f2:
            csv_reader = f2.readlines()
            for single_line in csv_reader:
                item = single_line.strip().split(".jpg ")
                token_path = os.path.join(self.text_feature_path, item[0] + '_token.npy')
                pm_path = os.path.join(self.text_feature_path, item[0] + '_pm.npy')
                if args.dataset == "MVSA" or args.dataset == "Food101" or args.dataset == "CUB":
                    visual_path = os.path.join(self.visual_feature_path, item[0] + ".jpg")
                else:
                    visual_path = os.path.join(self.visual_feature_path, item[0])

                if os.path.exists(token_path) and os.path.exists(visual_path):
                    data.append(item[0])
                    data2class[item[0]] = item[1]

        self.classes = sorted(classes)
        self.data2class = data2class

        self.av_files = []
        for item in data:
            self.av_files.append(item)
            self.drop.append(0)

        print('data load finish')
        length = len(self.av_files)

        # 模态级重采样（文本和视觉）
        gap_t = 1.0 - contribution_t
        gap_v = 1.0 - contribution_v

        if func == 'linear':
            difference = (abs(gap_t - gap_v) / 3 * 2) * alpha
        elif func == 'tanh':
            tanh = torch.nn.Tanh()
            difference = tanh(torch.tensor((abs(gap_t - gap_v) / 3 * 2) * alpha))
        elif func == 'square':
            difference = (abs(gap_t - gap_v) / 3 * 2) ** 1.5 * alpha
        
        resample_num = int(difference * length)
        sample_choice = np.random.choice(length, resample_num)

        for i in sample_choice:
            self.av_files.append(self.av_files[i])
            if gap_t > gap_v:
                self.drop.append(2)  # drop visual
            else:
                self.drop.append(1)  # drop text

        print('data resample finish')
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))

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
        self.preprocess_test = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.skip_norm = True
        self.noise = False
        self.norm_mean = -5.081
        self.norm_std = 4.4849

    def __len__(self):
        return len(self.av_files)
    
    def get_image(self, filename):
        img = Image.open(filename).convert('RGB')
        if self.mode == "train":
            image_tensor = self.preprocess_train(img)
        else:
            image_tensor = self.preprocess_test(img)
        return image_tensor

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        # Text
        token_path = os.path.join(self.text_feature_path, av_file + '_token.npy')
        pm_path = os.path.join(self.text_feature_path, av_file + '_pm.npy')
        tokenizer = np.load(token_path)
        padding_mask = np.load(pm_path)
        tokenizer = torch.tensor(tokenizer)
        padding_mask = torch.tensor(padding_mask)

        # Visual
        if self.dataset == "MVSA" or self.dataset == "Food101" or self.dataset == "CUB":
            image = self.get_image(os.path.join(self.visual_feature_path, av_file + ".jpg"))
        else:
            visual_path = os.path.join(self.visual_feature_path, av_file)
            allimages = os.listdir(visual_path)
            file_num = len(allimages)
            image = self.get_image(os.path.join(visual_path, allimages[int(file_num / 2)]))

        if self.skip_norm == False:
            tokenizer = (tokenizer - self.norm_mean) / (self.norm_std)

        if self.noise == True and self.mode == "train":
            tokenizer = tokenizer + torch.rand(tokenizer.shape[0], tokenizer.shape[1]) * np.random.rand() / 10
            tokenizer = torch.roll(tokenizer, np.random.randint(-1024, 1024), 0)

        label = self.classes.index(self.data2class[av_file])
        drop = self.drop[idx]

        return tokenizer, padding_mask, image, label, torch.LongTensor([idx]), drop


class IEMOCAPDataset_modality_level(Dataset):
    """IEMOCAP数据集的模态级重采样版本（三模态）"""
    
    def __init__(self, args, mode='train', contribution_a=0.33, contribution_v=0.33, contribution_t=0.34, 
                 alpha=1.0, func='linear'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode
        self.dataset = args.dataset
        self.drop = []
        
        if args.dataset == "IEMOCAP":
            self.data_root = '/data1/zhangxiaohui/IEMOCAP/'
            self.visual_feature_path = os.path.join(self.data_root, "visual", '{}_imgs/'.format(mode))
            self.text_feature_path = os.path.join(self.data_root, "text_token", '{}_token/'.format(mode))
            self.audio_feature_path = os.path.join(self.data_root, "audio", '{}_fbank/'.format(mode))
            self.stat_path = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/stat_iemo.txt"
            self.train_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_train_iemo.txt"
            self.test_txt = "/data1/zhangxiaohui/Multimodal-Learning-Adaptation/data/my_test_iemo.txt"

        with open(self.stat_path, "r") as f1:
            classes = f1.readlines()
        
        classes = [sclass.strip() for sclass in classes]

        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(csv_file, "r") as f2:
            csv_reader = f2.readlines()
            for single_line in csv_reader:
                item = single_line.strip().split(" [split|sign] ")
                item[0] = item[0].split(".mp4")[0]
                token_path = os.path.join(self.text_feature_path, item[0] + '_token.npy')
                visual_path = os.path.join(self.visual_feature_path, item[0])
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.npy')

                if os.path.exists(token_path) and os.path.exists(visual_path) and os.path.exists(audio_path):
                    data.append(item[0])
                    data2class[item[0]] = item[-1]

        self.classes = sorted(classes)
        self.data2class = data2class

        self.av_files = []
        for item in data:
            self.av_files.append(item)
            self.drop.append(0)

        print('data load finish')
        length = len(self.av_files)

        # 三模态重采样：选择贡献最小的模态进行drop
        contributions = {'audio': contribution_a, 'visual': contribution_v, 'text': contribution_t}
        min_modality = min(contributions, key=contributions.get)
        max_modality = max(contributions, key=contributions.get)
        
        gap_min = 1.0 - contributions[min_modality]
        gap_max = 1.0 - contributions[max_modality]

        if func == 'linear':
            difference = (abs(gap_min - gap_max) / 3 * 2) * alpha
        elif func == 'tanh':
            tanh = torch.nn.Tanh()
            difference = tanh(torch.tensor((abs(gap_min - gap_max) / 3 * 2) * alpha))
        elif func == 'square':
            difference = (abs(gap_min - gap_max) / 3 * 2) ** 1.5 * alpha
        
        resample_num = int(difference * length)
        sample_choice = np.random.choice(length, resample_num)

        drop_map = {'audio': 1, 'visual': 2, 'text': 3}
        for i in sample_choice:
            self.av_files.append(self.av_files[i])
            self.drop.append(drop_map[min_modality])

        print('data resample finish')
        print('# of files = %d ' % len(self.av_files))

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
        self.preprocess_test = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.norm_mean = -5.081
        self.norm_std = 4.4849

        if args.mask_percent or args.mask_percent == 0:
            samplenum = len(self.av_files)
            self.maskmatrix = random_mask(3, samplenum, args.mask_percent)

    def __len__(self):
        return len(self.av_files)
    
    def get_image(self, filename):
        img = Image.open(filename)
        if self.mode == "train":
            image_tensor = self.preprocess_train(img)
        else:
            image_tensor = self.preprocess_test(img)
        return image_tensor

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        # Text
        token_path = os.path.join(self.text_feature_path, av_file + '_token.npy')
        pm_path = os.path.join(self.text_feature_path, av_file + '_pm.npy')
        tokenizer = np.load(token_path)
        padding_mask = np.load(pm_path)
        tokenizer = torch.tensor(tokenizer)
        padding_mask = torch.tensor(padding_mask)

        # Visual
        visual_path = os.path.join(self.visual_feature_path, av_file)
        allimages = os.listdir(visual_path)
        image = self.get_image(os.path.join(visual_path, allimages[int(len(allimages) / 2)]))
        
        # Audio
        audio_path = os.path.join(self.audio_feature_path, av_file + '.npy')
        spectrogram = np.load(audio_path)
        spectrogram = torch.tensor(spectrogram)

        label = self.classes.index(self.data2class[av_file])
        drop = self.drop[idx]

        mask_seq = self.maskmatrix[idx]
        missing_index = torch.LongTensor(mask_seq)
        spectrogram = spectrogram * missing_index[0]
        image = image * missing_index[1]
        tokenizer = tokenizer * missing_index[2]
        padding_mask = padding_mask * missing_index[2]
        
        return tokenizer, padding_mask, image, spectrogram, label, torch.LongTensor([idx]), drop


class TVADataset_modality_level(Dataset):
    """TVA数据集的模态级重采样版本（三模态）"""
    
    def __init__(self, args, mode='train', pick_num=3, contribution_a=0.33, contribution_v=0.33, 
                 contribution_t=0.34, alpha=1.0, func='linear'):
        self.args = args
        self.mode = mode
        self.pick_num = pick_num
        self.drop = []
        
        self.sep_map = {
            "IEMOCAP3": ",",
            "URFUNNY": ",",
        }
        
        self._init_paths()
        self._init_classes()
        self._init_transforms()
        self._init_data_base()
        
        # 模态级重采样
        length = len(self.data)
        
        contributions = {'audio': contribution_a, 'visual': contribution_v, 'text': contribution_t}
        min_modality = min(contributions, key=contributions.get)
        max_modality = max(contributions, key=contributions.get)
        
        gap_min = 1.0 - contributions[min_modality]
        gap_max = 1.0 - contributions[max_modality]

        if func == 'linear':
            difference = (abs(gap_min - gap_max) / 3 * 2) * alpha
        elif func == 'tanh':
            tanh = torch.nn.Tanh()
            difference = tanh(torch.tensor((abs(gap_min - gap_max) / 3 * 2) * alpha))
        elif func == 'square':
            difference = (abs(gap_min - gap_max) / 3 * 2) ** 1.5 * alpha
        
        resample_num = int(difference * length)
        sample_choice = np.random.choice(length, resample_num)

        drop_map = {'audio': 1, 'visual': 2, 'text': 3}
        for i in sample_choice:
            self.data.append(self.data[i])
            self.drop.append(drop_map[min_modality])

        print('data resample finish')
        print(f"Final # of samples = {len(self.data)}")
    
    def _init_paths(self):
        if self.args.dataset == "IEMOCAP3":
            self.data_root = "/root/autodl-tmp/AME/data/dataset/IEMOCAP"
            self.audio_feature_path = os.path.join(self.data_root, "IEMOCAP_full_release", "fbank")
            self.visual_feature_path = os.path.join(self.data_root, "IEMOCAP_full_release", "IMAGE_KEPT_2_PER_SEC")
            self.text_feature_path = os.path.join(self.data_root, "IEMOCAP_full_release", "text_token")
            self.stat_path = os.path.join(self.data_root, "iemocap_stat.txt")
            
            if self.mode == 'train':
                self.data_file = os.path.join(self.data_root, "iemocap_train.txt")
            elif self.mode == 'val':
                self.data_file = os.path.join(self.data_root, "iemocap_val.txt")
            else:
                self.data_file = os.path.join(self.data_root, "iemocap_test.txt")
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")
    
    def _init_classes(self):
        with open(self.stat_path, "r") as f:
            self.classes = sorted([line.strip() for line in f])
    
    def _init_transforms(self):
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def _init_data_base(self):
        self.data = []
        self.data2class = {}
        self.audio_sid_path_map = {}
        self.visual_sid_path_map = {}
        self.text_token_path_map = {}
        self.text_pm_path_map = {}
        self.sid_all_imgs_map = {}
        
        sep = self.sep_map.get(self.args.dataset, ",")
        
        with open(self.data_file, "r", encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                item = [i.strip() for i in line.split(sep)]
                if len(item) < 2:
                    continue
                
                sid = item[0]
                label = item[1]
                
                audio_path = os.path.join(self.audio_feature_path, f"{sid}.npy")
                visual_path = os.path.join(self.visual_feature_path, sid)
                token_path = os.path.join(self.text_feature_path, f"{sid}_token.pt")
                pm_path = os.path.join(self.text_feature_path, f"{sid}_pm.pt")
                
                if not all([os.path.exists(p) for p in [audio_path, visual_path, token_path, pm_path]]):
                    continue
                
                try:
                    image_files = [f for f in os.listdir(visual_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    if not image_files:
                        continue
                    image_files.sort()
                except:
                    continue
                
                if label not in self.classes:
                    continue
                
                self.data.append(sid)
                self.data2class[sid] = label
                self.audio_sid_path_map[sid] = audio_path
                self.visual_sid_path_map[sid] = visual_path
                self.text_token_path_map[sid] = token_path
                self.text_pm_path_map[sid] = pm_path
                self.sid_all_imgs_map[sid] = image_files
                self.drop.append(0)
        
        print(f"Loaded {len(self.data)} samples for {self.mode} mode")
    
    def _load_visual_features(self, visual_path, allimages):
        file_num = len(allimages)
        if file_num == 0:
            raise RuntimeError(f"No images found in {visual_path}")
        
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
            token_path = self.text_token_path_map[sid]
            pm_path = self.text_pm_path_map[sid]
            
            tokenizer = torch.load(token_path, map_location='cpu')
            padding_mask = torch.load(pm_path, map_location='cpu')
            
            if tokenizer.dim() > 1:
                tokenizer = tokenizer.squeeze(0)
            if padding_mask.dim() > 1:
                padding_mask = padding_mask.squeeze(0)
            
            audio_path = self.audio_sid_path_map[sid]
            audio_feature = torch.from_numpy(np.load(audio_path).astype(np.float32))
            
            visual_path = self.visual_sid_path_map[sid]
            allimages = self.sid_all_imgs_map[sid]
            image_n = self._load_visual_features(visual_path, allimages)
            
            label = self.classes.index(self.data2class[sid])
            drop = self.drop[idx]
            
            return tokenizer, padding_mask, image_n, audio_feature, label, sid, drop
            
        except Exception as e:
            print(f"Error loading sample {sid} at index {idx}: {e}")
            dummy_token = torch.zeros(128, dtype=torch.long)
            dummy_mask = torch.zeros(128, dtype=torch.float)
            dummy_image = torch.zeros(3, self.pick_num, 224, 224)
            dummy_audio = torch.zeros(1024, 128)
            dummy_label = 0
            dummy_drop = 0
            return dummy_token, dummy_mask, dummy_image, dummy_audio, dummy_label, sid, dummy_drop
    
    def get_sample_info(self, idx):
        sid = self.data[idx]
        return {
            'sid': sid,
            'label': self.data2class[sid],
            'audio_path': self.audio_sid_path_map[sid],
            'visual_path': self.visual_sid_path_map[sid],
            'token_path': self.text_token_path_map[sid],
            'pm_path': self.text_pm_path_map[sid],
            'num_images': len(self.sid_all_imgs_map[sid]),
            'drop': self.drop[idx]
        }

def random_mask(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 3.2 of the paper
    :return: Sn [alldata_len, view_num]
    """
    # print (f'==== generate random mask ====')
    one_rate = 1-missing_rate      # missing_rate: 0.8; one_rate: 0.2

    if one_rate <= (1 / view_num): # 
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray() # only select one view [avoid all zero input]
        return view_preserve # [samplenum, viewnum=2] => one value set=1, others=0

    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num)) # [samplenum, viewnum=2] => all ones
        return matrix

    ## for one_rate between [1 / view_num, 1] => can have multi view input
    ## ensure at least one of them is avaliable 
    ## since some sample is overlapped, which increase difficulties
    error = 1
    while error >= 0.005:

        ## gain initial view_preserve
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray() # [samplenum, viewnum=2] => one value set=1, others=0

        ## further generate one_num samples
        one_num = view_num * alldata_len * one_rate - alldata_len  # left one_num after previous step
        ratio = one_num / (view_num * alldata_len)                 # now processed ratio
        # print (f'first ratio: {ratio}')
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int) # based on ratio => matrix_iter
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int)) # a: overlap number
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        # print (f'second ratio: {ratio}')
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        # print (f'third ratio: {ratio}')
        error = abs(one_rate - ratio)
        
    return matrix

# class CramedDataset_modality_level(Dataset):
#     """CREMA-D数据集的模态级重采样版本"""
    
#     def __init__(self, args, mode='train', contribution_a=0.5, contribution_v=0.5, alpha=1.0, func='linear'):
#         self.args = args
#         self.image = []
#         self.audio = []
#         self.label = []
#         self.drop = []  # 0=none, 1=audio, 2=visual
#         self.mode = mode

#         # 数据集基础路径配置
#         self.data_root = '/root/autodl-tmp/AME/data/dataset/CREMA-D/CREMA-D'
#         self.class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}
#         self.audio_feature_path = os.path.join(self.data_root, "AudioWAV")
#         self.train_csv = os.path.join(self.data_root, 'train.csv')
#         self.test_csv = os.path.join(self.data_root, 'test.csv')

#         # 加载原始数据
#         csv_file = self.train_csv if mode == 'train' else self.test_csv
#         with open(csv_file, encoding='UTF-8-sig') as f2:
#             csv_reader = csv.reader(f2)
#             for item in csv_reader:
#                 audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
#                 visual_path = os.path.join(self.data_root, f'Image-{self.args.fps:02d}-FPS', item[0])

#                 if os.path.exists(audio_path) and os.path.exists(visual_path):
#                     self.image.append(visual_path)
#                     self.audio.append(audio_path)
#                     self.label.append(self.class_dict[item[1]])
#                     self.drop.append(0)

#         print('data load finish')
#         length = len(self.image)

#         # 模态级重采样逻辑
#         gap_a = 1.0 - contribution_a
#         gap_v = 1.0 - contribution_v

#         if func == 'linear':
#             difference = (abs(gap_a - gap_v) / 3 * 2) * alpha
#         elif func == 'tanh':
#             tanh = torch.nn.Tanh()
#             difference = tanh(torch.tensor((abs(gap_a - gap_v) / 3 * 2) * alpha))
#         elif func == 'square':
#             difference = (abs(gap_a - gap_v) / 3 * 2) ** 1.5 * alpha
        
#         resample_num = int(difference * length)
#         sample_choice = np.random.choice(length, resample_num)

#         for i in sample_choice:
#             self.image.append(self.image[i])
#             self.audio.append(self.audio[i])
#             self.label.append(self.label[i])
#             if gap_a > gap_v:
#                 self.drop.append(2)  # drop visual
#             else:
#                 self.drop.append(1)  # drop audio

#         print('data resample finish')
#         print('# of files = %d ' % len(self.image))

#     def __len__(self):
#         return len(self.image)

#     def __getitem__(self, idx):
#         # 音频处理
#         samples, rate = librosa.load(self.audio[idx], sr=22050)
#         resamples = np.tile(samples, 3)[:22050 * 3]
#         resamples = np.clip(resamples, -1., 1.)
#         spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
#         spectrogram = np.log(np.abs(spectrogram) + 1e-7)

#         # 视觉处理
#         if self.mode == 'train':
#             transform = transforms.Compose([
#                 transforms.RandomResizedCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])
#         else:
#             transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ])

#         image_samples = os.listdir(self.image[idx])
#         image_samples.sort()

#         if len(image_samples) < self.args.fps:
#             select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=True)
#         else:
#             select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=False)
#         select_index.sort()

#         images = torch.zeros((self.args.fps, 3, 224, 224), dtype=torch.float32)
#         for i in range(self.args.fps):
#             img_path = os.path.join(self.image[idx], image_samples[select_index[i]])
#             img = Image.open(img_path).convert('RGB')
#             images[i] = transform(img)

#         images = torch.permute(images, (1, 0, 2, 3))
#         label = self.label[idx]
#         drop = self.drop[idx]
#         sid = self.audio[idx].split('/')[-1].split('.')[0]

#         return spectrogram, images, label, sid, drop


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
        print(f"contribution is {contribution}")
        # 样本级重采样逻辑（基于每个样本的贡献度）
        if contribution is not None:
            # added_count = 0
            added_count = 0
            contrib_stats = {'audio_high': 0, 'audio_mid': 0, 'audio_low': 0, 
                           'visual_high': 0, 'visual_mid': 0, 'visual_low': 0}
            for i in range(length):
                sid = self.data[i]
                # contrib_a, contrib_v = contribution.get(sid, (0.0, 0.0))
                contrib_a, contrib_v = contribution[sid]
                
                # 调试信息：打印前几个样本的贡献度
                if i < 3:
                    print(f"Sample {i} (sid={sid}): contrib_a={contrib_a:.4f}, contrib_v={contrib_v:.4f}")
                
                # 根据音频贡献度复制样本并drop视觉
                if 0.4 < contrib_a < 1:  # 音频贡献度中等
                    for _ in range(1):
                        self.data.append(self.data[i])
                        self.drop.append(1)  # drop visual
                    contrib_stats['audio_high'] += 1
                elif -0.1 < contrib_a < 0.4:  # 音频贡献度较低
                    for _ in range(2):
                        self.data.append(self.data[i])
                        self.drop.append(1)
                    contrib_stats['audio_mid'] += 1
                elif contrib_a < -0.1:  # 音频贡献度很低
                    for _ in range(3):
                        self.data.append(self.data[i])
                        self.drop.append(1)
                    contrib_stats['audio_low'] += 1
                
                # 根据视觉贡献度复制样本并drop音频
                if 0.4 < contrib_v < 1:
                    for _ in range(1):
                        self.data.append(self.data[i])
                        self.drop.append(2)  # drop audio
                    contrib_stats['visual_high'] += 1
                elif -0.1 < contrib_v < 0.4:
                    for _ in range(2):
                        self.data.append(self.data[i])
                        self.drop.append(2)
                    contrib_stats['visual_mid'] += 1
                elif contrib_v < -0.1:
                    for _ in range(3):
                        self.data.append(self.data[i])
                        self.drop.append(2)
                        added_count += 1
                    contrib_stats['visual_mid'] += 1
            
            print(f'data resample finish: added {added_count} samples, total {len(self.data)} samples')
            print(f'  Original samples: {length}')
            print(f'  Added samples: {added_count}')
            print(f'  Total samples: {len(self.data)}')
            print(f'  Contribution stats: Audio(high={contrib_stats["audio_high"]}, mid={contrib_stats["audio_mid"]}, low={contrib_stats["audio_low"]}), '
                  f'Visual(high={contrib_stats["visual_high"]}, mid={contrib_stats["visual_mid"]}, low={contrib_stats["visual_low"]})')
        
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
        
        return spectrogram, images, label, datum, drop, _


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
                sid = self.audio[i].split('/')[-1].split('.')[0]
                contrib_a, contrib_v = contribution.get(sid, (0.0, 0.0))
                
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
                sid = self.av_files[i]
                contrib_t, contrib_v = contribution.get(sid, (0.0, 0.0))
                
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
                sid = self.data[i]
                contrib_t, contrib_v, contrib_a = self.contribution.get(sid, (0.0, 0.0, 0.0))
                
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
        

