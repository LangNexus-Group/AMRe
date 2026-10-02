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

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))

# --------------------------CramedDataset --------------------------
class CramedDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []  # 存储视觉帧文件夹路径
        self.audio = []  # 存储音频文件路径
        self.label = []  # 存储样本标签
        self.mode = mode  # 训练/测试模式

        # 数据集基础路径配置
        self.data_root = '/root/autodl-tmp/AME/data/dataset/CREMA-D/CREMA-D'
        self.class_dict = {'NEU': 0, 'HAP': 1, 'SAD': 2, 'FEA': 3, 'DIS': 4, 'ANG': 5}  # 情感标签映射
        self.audio_feature_path = os.path.join(self.data_root, "AudioWAV")  # 音频文件根目录
        self.train_csv = os.path.join(self.data_root, 'train.csv')  # 训练集划分文件
        self.test_csv = os.path.join(self.data_root, 'test.csv')    # 测试集划分文件

        # 根据模式加载对应CSV文件
        csv_file = self.train_csv if mode == 'train' else self.test_csv
        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                # 构造音频路径（.wav格式）和视觉路径（按FPS区分的帧文件夹）
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
                visual_path = os.path.join(self.data_root, f'Image-{self.args.fps:02d}-FPS', item[0])

                # 仅保留音频和视觉文件均存在的有效样本
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(self.class_dict[item[1]])

    def __len__(self):
        # 返回样本总数（以视觉路径列表长度为准，确保音频/视觉/标签数量一致）
        return len(self.image)

    def __getitem__(self, idx):
        # -------------------------- 音频处理（保持原逻辑不变）--------------------------
        # 加载音频（采样率22050）
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        # 统一音频时长为3秒（不足则重复拼接，过长则截断）
        resamples = np.tile(samples, 3)[:22050 * 3]
        # 音频幅度钳位（限制在[-1, 1]范围，避免异常值）
        resamples = np.clip(resamples, -1., 1.)
        # 生成STFT谱图并对数压缩（增强低幅度信号区分度）
        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)  # 加1e-7避免log(0)

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

        # 1. 获取当前样本的所有帧图像文件名，并按字母/数字排序（确保帧的时间顺序）
        image_samples = os.listdir(self.image[idx])
        image_samples.sort()  # 关键：避免os.listdir返回顺序混乱导致帧顺序错误

        # 2. 随机选择self.args.fps帧（不重复），并按时间顺序排序
        # 若帧数量不足fps，允许重复采样（避免报错）
        if len(image_samples) < self.args.fps:
            select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=True)
        else:
            select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=False)
        select_index.sort()  # 排序索引：确保采样的帧按时间顺序排列，符合时序逻辑

        # 3. 加载并处理选中的帧（核心修复：使用select_index[i]而非i作为索引）
        images = torch.zeros((self.args.fps, 3, 224, 224), dtype=torch.float32)
        for i in range(self.args.fps):
            # 修复点：用随机生成的select_index[i]选择图像，而非固定取前fps张
            img_path = os.path.join(self.image[idx], image_samples[select_index[i]])
            # 加载图像并转为RGB通道（避免灰度图导致的通道数异常）
            img = Image.open(img_path).convert('RGB')
            # 应用图像变换
            images[i] = transform(img)

        # 调整维度顺序：从[fps, 3, 224, 224]转为[3, fps, 224, 224]（通道优先，适配模型输入）
        images = torch.permute(images, (1, 0, 2, 3))

        # -------------------------- 标签与样本ID处理（保持原逻辑不变）--------------------------
        label = self.label[idx]  # 样本标签（整数）
        # 提取样本ID（从音频路径中截取：如"xxx.wav"→"xxx"）
        sid = self.audio[idx].split('/')[-1].split('.')[0]

        # 返回格式：音频谱图（np.ndarray）、视觉帧（torch.Tensor）、标签（int）、样本ID（str）
        return spectrogram, images, label, sid

# =================KSDataset=======================
class KSDataset(nn.Module):
    def __init__(self, args, mode='train'):
        super().__init__()
        self.args = args

        self.labels = []
        self.videos = []
        self.audios = []
        data_path = '/root/autodl-tmp/AME/data/dataset/KSDataset/kinect_sound'
        # 处理class.txt文件，获取类别列表
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
        
        # 处理类别与标签的映射字典
        labels = range(len(class_list))
        data_dict = zip(class_list, labels)
        data_dict = dict(data_dict)

        self.mode = mode
        if self.mode == 'train':
            visual_data_path = os.path.join(data_path, 'visual', 'train_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'train')
        elif self.mode == 'test':
            visual_data_path = os.path.join(data_path, 'visual', 'val_img/Image-01-FPS')
            audio_data_path = os.path.join(data_path, 'audio', 'test')

        remove_list = []  # 移除损坏视频

        # 遍历每个类别，收集视频和音频路径及其对应标签
        for class_name in class_list:
            visual_class_path = os.path.join(visual_data_path, class_name)
            audio_class_path = os.path.join(audio_data_path, class_name)

            video_list = os.listdir(visual_class_path)
            video_list.sort()

            audio_list = os.listdir(audio_class_path)
            audio_list.sort()

            for video in video_list:
                # i+=1
                video_path = os.path.join(visual_class_path, video)

                if len(listdir_nohidden(video_path)) < 3:
                    # print(video_path)
                    remove_list.append(video)
                    continue

                self.videos.append(video_path)
                self.labels.append(data_dict[class_name])

            for audio in audio_list:
                if audio in remove_list:
                    print(audio)
                    continue
                audio_path = os.path.join(audio_class_path, audio)
                self.audios.append(audio_path)

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
        # print(len(image_samples))
        select_index = np.random.choice(len(image_samples), size=self.args.use_video_frames, replace=False)
        select_index.sort()
        images = torch.zeros((self.args.use_video_frames, 3, 224, 224))
        for i in range(self.args.use_video_frames):
            try:
                img = Image.open(image_samples[i]).convert('RGB')
            except Exception as e:
                print(e)
                print(image_samples[i])
                continue

            bt = time.time()
            img = transform(img)
            et = time.time()
            # print(et-bt)
            images[i] = img

        images = torch.permute(images, (1, 0, 2, 3))

        # label
        label = self.labels[idx]
        sid = self.audios[idx].split('/')[-1].split('.')[0]
        # print(label)
        return spectrogram, images, label,sid
# ================AVEDataset=======================
class AVEDataset(Dataset):

    def __init__(self, args, mode='train'):
        self.args = args
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        classes = []
        self.data_root = '/root/autodl-tmp/AME/data/dataset/AVE/AVE_Dataset'

        self.visual_feature_path = self.data_root
        self.audio_feature_path = os.path.join(self.data_root,'Audio-1004-SE')

        self.train_txt = os.path.join(self.data_root, 'trainSet.txt')
        self.test_txt = os.path.join(self.data_root, 'testSet.txt')
        self.val_txt = os.path.join(self.data_root, 'valSet.txt')

        if mode == 'train':
            txt_file = self.train_txt
        elif mode == 'test':
            txt_file = self.test_txt
        else:
            txt_file = self.val_txt
        print(f"mode is {mode}")
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
                # print(f"item")
                audio_path = os.path.join(self.audio_feature_path, item[1] + '.pkl')
                visual_path = os.path.join(self.visual_feature_path, 'Image-01-FPS-SE',item[1])
                                           
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    if os.stat(audio_path).st_size < 200:
                        print(audio_path)
                        continue
                    if audio_path not in self.audio:
                        self.image.append(visual_path)
                        self.audio.append(audio_path)
                        self.label.append(class_dict[item[0]])
                else:
                    continue

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
        # image_samples = os.listdir(self.image[idx])

        # images = torch.zeros((self.args.num_frame, 3, 224, 224))
        # for i in range(self.args.num_frame):
        #     # for i, n in enumerate(select_index):
        #     img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
        #     # print(os.path.join(self.image[idx], image_samples[i]))
        #     img = transform(img)
        #     images[i] = img

        # images = torch.permute(images, (1, 0, 2, 3))
        # 1. 获取当前样本的所有帧图像文件名，并按字母/数字排序（确保帧的时间顺序）
        image_samples = os.listdir(self.image[idx])
        image_samples.sort()  # 关键：避免os.listdir返回顺序混乱导致帧顺序错误

        # 2. 随机选择self.args.fps帧（不重复），并按时间顺序排序
        # 若帧数量不足fps，允许重复采样（避免报错）
        if len(image_samples) < self.args.fps:
            select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=True)
        else:
            select_index = np.random.choice(len(image_samples), size=self.args.fps, replace=False)
        select_index.sort()  # 排序索引：确保采样的帧按时间顺序排列，符合时序逻辑

        # 3. 加载并处理选中的帧（核心修复：使用select_index[i]而非i作为索引）
        images = torch.zeros((self.args.fps, 3, 224, 224), dtype=torch.float32)
        for i in range(self.args.fps):
            # 修复点：用随机生成的select_index[i]选择图像，而非固定取前fps张
            img_path = os.path.join(self.image[idx], image_samples[select_index[i]])
            # 加载图像并转为RGB通道（避免灰度图导致的通道数异常）
            img = Image.open(img_path).convert('RGB')
            # 应用图像变换
            images[i] = transform(img)

        # 调整维度顺序：从[fps, 3, 224, 224]转为[3, fps, 224, 224]（通道优先，适配模型输入）
        images = torch.permute(images, (1, 0, 2, 3))
        # label
        label = self.label[idx]

        if spectrogram.shape != (257, 1004):
            print(self.audio[idx])
            print(spectrogram.shape)
            print("11111111111111")
        if images.shape != torch.Size([3, self.args.fps, 224, 224]):
            print(images.shape)
            print("22222222222")

        # print(spectrogram.shape,images.shape)
        sid = self.audio[idx].split('/')[-1].split('.')[0]
        # print(f"Data sid is {sid}")
        
        return spectrogram, images, label, sid

# ================Food101 and MVSA Dataset=======================
class M3AEDataset(Dataset):

    def __init__(self, args, mode='train'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode
        self.dataset = args.dataset
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
                # pdb.set_trace()
                if os.path.exists(token_path) and os.path.exists(visual_path):
                    data.append(item[0])
                    data2class[item[0]] = item[1]
                else:
                    if not os.path.exists(token_path):
                        print(f"token not have {item}")
                    if not os.path.exists(visual_path):
                        print(f"visual not have {item}")
                    continue

        self.classes = sorted(classes)

        print(self.classes)
        self.data2class = data2class

        self.av_files = []
        for item in data:
            self.av_files.append(item)
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))
        # self.preprocess = transforms.Compose([
        #     transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.4850, 0.4560, 0.4060],std=[0.2290, 0.2240, 0.2250])
        #     ])
        self.preprocess_train = create_transform(
                input_size = 224,
                is_training=True,
                color_jitter = True,
                auto_augment = None,
                interpolation = "bicubic",
                re_prob = 0,
                re_mode = 0,
                re_count = "const",
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225),
            )
        # self.preprocess_train = transforms.Compose(
        #         [
        #             transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        #             transforms.CenterCrop(224),
        #             transforms.ToTensor(),
        #             # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #         ]
        #     )
        self.preprocess_test = transforms.Compose(
                [
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        self.skip_norm = True
        self.noise = False
        self.norm_mean = -5.081
        self.norm_std = 4.4849
        

    def __len__(self):
        return len(self.av_files)
    
    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = Image.open(filename).convert('RGB')
            if self.mode == "train":
                image_tensor = self.preprocess_train(img)
            else:
                image_tensor = self.preprocess_test(img)
            return image_tensor
        else:
            img1 = Image.open(filename).convert('RGB')
            image_tensor1 = self.preprocess(img1)

            img2 = Image.open(filename2).convert('RGB')
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
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
        # if idx == 1422:
        #     print(f"token_path is {token_path}")
        # Visual
        if self.dataset == "MVSA" or self.dataset == "Food101" or self.dataset == "CUB":
            image = self.get_image(os.path.join(self.visual_feature_path, av_file + ".jpg"))
        else:
            visual_path = os.path.join(self.visual_feature_path, av_file)
            allimages = os.listdir(visual_path)
            file_num = len(allimages)
            image = self.get_image(os.path.join(visual_path, allimages[int(file_num / 2)]))
        # normalize the input for both training and test
        if self.skip_norm == False:
            tokenizer = (tokenizer - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True and self.mode == "train" and self.augnois:
            tokenizer = tokenizer + torch.rand(tokenizer.shape[0], tokenizer.shape[1]) * np.random.rand() / 10
            tokenizer = torch.roll(tokenizer, np.random.randint(-1024, 1024), 0)

        label = self.classes.index(self.data2class[av_file])
        # print(f"image shape is {image.shape}")
        return tokenizer, padding_mask, image, label, torch.LongTensor([int(av_file)])
    
# ================IEMOCAPDataset=======================
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

class IEMOCAPDataset(Dataset):

    def __init__(self, args, mode='train'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode
        # self.augnois = args.cav_augnois
        self.dataset = args.dataset
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
                pm_path = os.path.join(self.text_feature_path, item[0] + '_pm.npy')
                visual_path = os.path.join(self.visual_feature_path, item[0])
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.npy')
                # pdb.set_trace()
                if os.path.exists(token_path) and os.path.exists(visual_path) and os.path.exists(audio_path):
                    data.append(item[0])
                    data2class[item[0]] = item[-1]
                else:
                    continue

        self.classes = sorted(classes)

        print(self.classes)
        self.data2class = data2class

        self.av_files = []
        for item in data:
            self.av_files.append(item)
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))

        self.preprocess_train = create_transform(
                input_size = 224,
                is_training=True,
                color_jitter = True,
                auto_augment = None,
                interpolation = "bicubic",
                re_prob = 0,
                re_mode = 0,
                re_count = "const",
                mean = (0.485, 0.456, 0.406),
                std = (0.229, 0.224, 0.225),
            )
        self.preprocess_test = transforms.Compose(
                [
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        self.norm_mean = -5.081
        self.norm_std = 4.4849

        if args.mask_percent or args.mask_percent == 0:
            samplenum = len(self.av_files)
            # print (f'using random initialized mask!!')
            # acoustic_mask = (np.random.rand(samplenum, 1) > self.mask_rate).astype(int)
            # vision_mask = (np.random.rand(samplenum, 1) > self.mask_rate).astype(int)
            # lexical_mask = (np.random.rand(samplenum, 1) > self.mask_rate).astype(int)
            # self.maskmatrix = np.concatenate((acoustic_mask, vision_mask, lexical_mask), axis=1)
            self.maskmatrix = random_mask(3, samplenum, args.mask_percent) # [samplenum, view_num]
            # pdb.set_trace()
        

    def __len__(self):
        return len(self.av_files)
    
    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = Image.open(filename)
            if self.mode == "train":
                image_tensor = self.preprocess_train(img)
            else:
                image_tensor = self.preprocess_test(img)
            return image_tensor
        else:
            img1 = Image.open(filename)
            image_tensor1 = self.preprocess(img1)

            img2 = Image.open(filename2)
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
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
        # image = self.get_image(os.path.join(self.visual_feature_path, av_file + ".jpg"))
        # normalize the input for both training and test
        visual_path = os.path.join(self.visual_feature_path, av_file)
        allimages = os.listdir(visual_path)
        image = self.get_image(os.path.join(visual_path, allimages[int(len(allimages) / 2)]))
        # file_num = len(allimages)
        # pick_num = 1
        # seg = int(file_num / pick_num)
        # image_arr = []

        # for i in range(pick_num):
        #     tmp_index = int(seg * i / 2)
        #     # image = Image.open(os.path.join(visual_path, allimages[tmp_index])).convert('RGB')
        #     # image = transform(image)
        #     image = self.get_image(os.path.join(visual_path, allimages[tmp_index]))
        #     image = image.unsqueeze(1).float()
        #     image_arr.append(image)
        #     if i == 0:
        #         image_n = copy.copy(image_arr[i])
        #     else:
        #         image_n = torch.cat((image_n, image_arr[i]), 1)
        
        # Audio
        audio_path = os.path.join(self.audio_feature_path, av_file + '.npy')
        spectrogram = np.load(audio_path)
        spectrogram = torch.tensor(spectrogram)

        label = self.classes.index(self.data2class[av_file])

        mask_seq = self.maskmatrix[idx]
        missing_index = torch.LongTensor(mask_seq)
        # print(missing_index, missing_index.shape)
        # pdb.set_trace()
        spectrogram = spectrogram * missing_index[0]
        image = image * missing_index[1]
        tokenizer = tokenizer * missing_index[2]
        padding_mask = padding_mask * missing_index[2]
        
        return tokenizer, padding_mask, image, spectrogram, label, torch.LongTensor([int(av_file)])


class TVADataset(Dataset):
    """
    独立的TVA (Text-Visual-Audio) 数据集类
    支持三种模态：文本、视觉、音频
    """
    
    def __init__(self, args, mode='train', pick_num=3):
        self.args = args
        self.mode = mode
        self.pick_num = pick_num
        
        # 分隔符映射
        self.sep_map = {
            "IEMOCAP3": ",",
            "URFUNNY": ",",
        }
        
        # 初始化路径配置
        self._init_paths()
        
        # 初始化类别
        self._init_classes()
        
        # 初始化数据变换
        self._init_transforms()
        
        # 初始化数据
        self._init_data()
    
    def _init_paths(self):
        """初始化数据路径"""
        # 根据您的实际路径配置进行调整
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
            else:  # test
                self.data_file = os.path.join(self.data_root, "iemocap_test.txt")
                
        elif self.args.dataset == "URFUNNY":
            self.data_root = "/path/to/URFUNNY"
            self.audio_feature_path = os.path.join(self.data_root, "audio_features")
            self.visual_feature_path = os.path.join(self.data_root, "visual_features")
            self.text_feature_path = os.path.join(self.data_root, "text_features")
            self.stat_path = os.path.join(self.data_root, "stat.txt")
            
            if self.mode == 'train':
                self.data_file = os.path.join(self.data_root, "train.txt")
            elif self.mode == 'val':
                self.data_file = os.path.join(self.data_root, "val.txt")
            else:  # test
                self.data_file = os.path.join(self.data_root, "test.txt")
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")
    
    def _init_classes(self):
        """初始化类别信息"""
        if not os.path.exists(self.stat_path):
            raise FileNotFoundError(f"Stat file not found: {self.stat_path}")
            
        with open(self.stat_path, "r") as f:
            self.classes = sorted([line.strip() for line in f])
        
        print(f"Loaded {len(self.classes)} classes: {self.classes}")
    
    def _init_transforms(self):
        """初始化图像变换"""
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
    
    def _init_data(self):
        """初始化数据索引"""
        self.data = []
        self.data2class = {}
        self.audio_sid_path_map = {}
        self.visual_sid_path_map = {}
        self.text_token_path_map = {}
        self.text_pm_path_map = {}
        self.sid_all_imgs_map = {}
        
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        sep = self.sep_map.get(self.args.dataset, ",")
        
        with open(self.data_file, "r", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 解析行数据
                    item = [i.strip() for i in line.split(sep)]
                    if len(item) < 2:
                        print(f"Line {line_num}: Invalid format, skipping: {line}")
                        continue
                    
                    sid = item[0]   # sample id
                    label = item[1] # label
                    
                    # 构建文件路径
                    audio_path = os.path.join(self.audio_feature_path, f"{sid}.npy")
                    visual_path = os.path.join(self.visual_feature_path, sid)
                    token_path = os.path.join(self.text_feature_path, f"{sid}_token.pt")
                    pm_path = os.path.join(self.text_feature_path, f"{sid}_pm.pt")
                    
                    # 检查所有文件是否存在
                    missing_files = []
                    if not os.path.exists(audio_path):
                        missing_files.append(f"audio: {audio_path}")
                    if not os.path.exists(visual_path):
                        missing_files.append(f"visual: {visual_path}")
                    if not os.path.exists(token_path):
                        missing_files.append(f"token: {token_path}")
                    if not os.path.exists(pm_path):
                        missing_files.append(f"pm: {pm_path}")
                    
                    if missing_files:
                        print(f"Line {line_num}: Missing files for {sid}: {', '.join(missing_files)}")
                        continue
                    
                    # 检查视觉目录中的图像
                    try:
                        image_files = [f for f in os.listdir(visual_path) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                        if not image_files:
                            print(f"Line {line_num}: No image files found in {visual_path}")
                            continue
                        
                        # 排序确保一致性
                        image_files.sort()
                        
                    except Exception as e:
                        print(f"Line {line_num}: Error reading visual directory {visual_path}: {e}")
                        continue
                    
                    # 验证标签是否在类别中
                    if label not in self.classes:
                        print(f"Line {line_num}: Unknown label '{label}' for sample {sid}")
                        continue
                    
                    # 添加到数据集
                    self.data.append(sid)
                    self.data2class[sid] = label
                    self.audio_sid_path_map[sid] = audio_path
                    self.visual_sid_path_map[sid] = visual_path
                    self.text_token_path_map[sid] = token_path
                    self.text_pm_path_map[sid] = pm_path
                    self.sid_all_imgs_map[sid] = image_files
                    
                except Exception as e:
                    print(f"Line {line_num}: Error processing line '{line}': {e}")
                    continue
        
        print(f"Loaded {len(self.data)} samples for {self.mode} mode")
        
        # 统计类别分布
        class_counts = {}
        for sid in self.data:
            label = self.data2class[sid]
            class_counts[label] = class_counts.get(label, 0) + 1
        
        print("Class distribution:")
        for cls, count in sorted(class_counts.items()):
            print(f"  {cls}: {count}")
    
    def _load_visual_features(self, visual_path, allimages):
        """
        加载视觉特征，支持多帧采样
        
        Args:
            visual_path (str): 图像文件夹路径
            allimages (list): 图像文件名列表
            
        Returns:
            torch.Tensor: 形状为 (C, pick_num, H, W) 的张量
        """
        file_num = len(allimages)
        
        if file_num == 0:
            raise RuntimeError(f"No images found in {visual_path}")
        
        # 计算采样间隔
        if file_num >= self.pick_num:
            seg = file_num / self.pick_num
            indices = [min(int(seg * i), file_num - 1) for i in range(self.pick_num)]
        else:
            # 如果图像数量少于需要的数量，进行循环采样
            indices = [i % file_num for i in range(self.pick_num)]
        
        image_arr = []
        for idx in indices:
            img_path = os.path.join(visual_path, allimages[idx])
            try:
                with Image.open(img_path) as img:
                    image = self.transform(img.convert('RGB')).unsqueeze(1).float()
                image_arr.append(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # 创建零张量作为备用
                dummy_img = torch.zeros(3, 1, 224, 224)
                image_arr.append(dummy_img)
        
        return torch.cat(image_arr, 1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Returns:
            tuple: (tokenizer, padding_mask, image_n, audio_feature, label, sid)
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        sid = self.data[idx]
        
        try:
            # 加载文本特征
            token_path = self.text_token_path_map[sid]
            pm_path = self.text_pm_path_map[sid]
            
            tokenizer = torch.load(token_path, map_location='cpu')
            padding_mask = torch.load(pm_path, map_location='cpu')
            
            # 确保维度正确
            if tokenizer.dim() > 1:
                tokenizer = tokenizer.squeeze(0)
            if padding_mask.dim() > 1:
                padding_mask = padding_mask.squeeze(0)
            
            # 加载音频特征
            audio_path = self.audio_sid_path_map[sid]
            audio_feature = torch.from_numpy(np.load(audio_path).astype(np.float32))
            
            # 加载视觉特征
            visual_path = self.visual_sid_path_map[sid]
            allimages = self.sid_all_imgs_map[sid]
            image_n = self._load_visual_features(visual_path, allimages)
            
            # 获取标签
            label = self.classes.index(self.data2class[sid])
            
            return tokenizer, padding_mask, image_n, audio_feature, label, sid
            
        except Exception as e:
            print(f"Error loading sample {sid} at index {idx}: {e}")
            # 返回错误时的默认值
            dummy_token = torch.zeros(128, dtype=torch.long)
            dummy_mask = torch.zeros(128, dtype=torch.float)
            dummy_image = torch.zeros(3, self.pick_num, 224, 224)
            dummy_audio = torch.zeros(1024, 128)
            dummy_label = 0
            return dummy_token, dummy_mask, dummy_image, dummy_audio, dummy_label, sid
    
    def get_sample_info(self, idx):
        """获取样本信息（用于调试）"""
        sid = self.data[idx]
        return {
            'sid': sid,
            'label': self.data2class[sid],
            'audio_path': self.audio_sid_path_map[sid],
            'visual_path': self.visual_sid_path_map[sid],
            'token_path': self.text_token_path_map[sid],
            'pm_path': self.text_pm_path_map[sid],
            'num_images': len(self.sid_all_imgs_map[sid])
        }