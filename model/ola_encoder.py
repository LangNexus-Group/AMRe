import os
os.environ['LOWRES_RESIZE'] = '384x32'
os.environ['HIGHRES_BASE'] = '0x32'
os.environ['VIDEO_RESIZE'] = "0x64"
os.environ['VIDEO_MAXRES'] = "480"
os.environ['VIDEO_MINRES'] = "288"
os.environ['MAXRES'] = '1536'
os.environ['MINRES'] = '0'
os.environ['FORCE_NO_DOWNSAMPLE'] = '1'
os.environ['LOAD_VISION_EARLY'] = '1'
os.environ['PAD2STRIDE'] = '1'

import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加到sys.path
sys.path.insert(0, current_dir)

# sys.path.append("/root/autodl-tmp/AMRe/model/ola") 

import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import librosa
from decord import VideoReader, cpu
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import whisper
from torch.amp import autocast
from typing import Optional

from .fusion_model import ConcatFusion, SumFusion

from .ola.conversation import conv_templates, SeparatorStyle
from .ola.model.builder import load_pretrained_model
from .ola.datasets.preprocess import tokenizer_image_token, tokenizer_speech_token
from .ola.mm_utils import process_anyres_video, process_anyres_highres_image
from .ola.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX

class OLA_Classifier(nn.Module):
    def __init__(self, args):
        super(OLA_Classifier,self).__init__()
        fusion = args.fusion_method
        self.args = args
        self.outdim = args.num_classes
        self.conv_mode = "qwen_1_5"
        self.dataset = args.dataset
        if fusion == "concat":
            print(f"***Use Concat Fusion Model***")
            self.fusion_model = ConcatFusion(args)
        elif fusion == 'sum':
            print(f"***Use Sum Fusion Model***")
            self.fusion_model = SumFusion(args)
        # OLA settting 
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model("/root/autodl-tmp/AMRe/model/Ola-7b", None)
        self.image_processor.do_resize = False
        self.image_processor.do_center_crop = False
        self.use_frame_dir = True
        # print(self.model.device)

        self.tokenizer.add_tokens("[CLASS]")
        self.class_token_id = self.tokenizer.convert_tokens_to_ids("[CLASS]")
        self.model.resize_token_embeddings(len(self.tokenizer))
        # frozen model parameters in training
        for param in self.model.parameters():
            param.requires_grad = False
        self.model = self.model.bfloat16()
        
        self.pad_token_ids = 151669
        
        # Dataset setting
        if self.dataset == "CREMAD":
            self.data_root = "/root/autodl-tmp/AMRe/data/Dataset/CREMA-D/CREMA-D"
            self.visual_feature_path = os.path.join(self.data_root, "VideoFlash/")
            self.frame_feature_path = os.path.join(self.data_root, "Image-01-FPS")
            self.use_frame_dir = os.path.isdir(self.frame_feature_path)
            self.audio_feature_path = os.path.join(self.data_root, "AudioWAV/")
            self.OLA_dir = '/root/autodl-tmp/AMRe/data/Dataset/CREMA-D/OLA_Feature'
        elif self.dataset == "AVE":
            self.data_root = "/root/autodl-tmp/AMRe/data/Dataset/AVE/AVE_Dataset"
            self.visual_feature_path = os.path.join(self.data_root, "Visual/")
            self.frame_feature_path = os.path.join(self.data_root, "Image-01-FPS-SE")
            self.use_frame_dir = os.path.isdir(self.frame_feature_path)
            self.audio_feature_path = "/root/autodl-tmp/AMRe/data/Dataset/AVE/new_AVE_Dataset/AVE_Dataset/Audios"
            self.OLA_dir = '/root/autodl-tmp/AMRe/data/Dataset/AVE/AVE_Dataset/OLA_Feature'
        elif self.dataset == "MVSA":
            self.data_root = '/root/autodl-tmp/AMRe/data/Dataset/MVSA/MVSA_Single'
            self.OLA_dir = os.path.join(self.data_root, 'OLA_Feature')

    def load_audio(self, audio_file_name):
        speech_wav, samplerate = librosa.load(audio_file_name, sr=16000)
        if len(speech_wav.shape) > 1:
            speech_wav = speech_wav[:, 0]
        speech_wav = speech_wav.astype(np.float32)
        CHUNK_LIM = 480000
        SAMPLE_RATE = 16000
        speechs = []
        speech_wavs = []

        if len(speech_wav) <= CHUNK_LIM:
            speech = whisper.pad_or_trim(speech_wav)
            speech_wav = whisper.pad_or_trim(speech_wav)
            speechs.append(speech)
            speech_wavs.append(torch.from_numpy(speech_wav).unsqueeze(0))
        else:
            for i in range(0, len(speech_wav), CHUNK_LIM):
                chunk = speech_wav[i : i + CHUNK_LIM]
                if len(chunk) < CHUNK_LIM:
                    chunk = whisper.pad_or_trim(chunk)
                speechs.append(chunk)
                speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
        mels = []
        for chunk in speechs:
            chunk = whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0)
            mels.append(chunk)

        mels = torch.cat(mels, dim=0)
        speech_wavs = torch.cat(speech_wavs, dim=0)
        if mels.shape[0] > 25:
            mels = mels[:25]
            speech_wavs = speech_wavs[:25]

        speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
        speech_chunks = torch.LongTensor([mels.shape[0]])
        return mels, speech_length, speech_chunks, speech_wavs
    
    
    def pkl_spec_to_mel(self, pkl_path, sr=16000, n_fft=512, n_mels=128):
        spec = pickle.load(open(pkl_path, "rb"))  # (257, T)
        spec = np.asarray(spec, dtype=np.float32)
    
        # log -> linear
        spec = np.exp(spec) - 1e-7
    
        mel_filter = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        mel = mel_filter @ spec  # (128, T)
        mel = np.log(mel + 1e-7)
    
        mel = torch.from_numpy(mel.T).unsqueeze(0)  # (1, T, 128)
        speech_length = torch.LongTensor([mel.shape[1]])
        speech_chunks = torch.LongTensor([mel.shape[0]])
        speech_wavs = torch.zeros((mel.shape[0], 480000))
        return mel, speech_length, speech_chunks, speech_wavs
    def process_video(self, file_name):
        vr = VideoReader(file_name, ctx=cpu(0))
        # 1. 均匀采样 2 帧
        uniform_sampled_frames = np.linspace(0, len(vr) - 1, 2, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        
        # 2. 提取这 2 帧
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        video = [Image.fromarray(frame) for frame in spare_frames]
        
        video_processed = []
        for frame in video:
            # 3. 处理每一帧
            processed_frame = process_anyres_video(frame, self.image_processor)
            # processed_frame 形状通常是 [Num_Patches, C, H, W]
            video_processed.append(processed_frame) # 直接 append
        
        # 4. 拼接

        video_final = torch.cat(video_processed, dim=0).bfloat16().to("cuda").unsqueeze(0)
        

        
        video_data = ((video_final, video_final), (384, 384), "video")
        return video_data

    def process_frames(self, frame_dir, num_frames=2):
        frame_files = sorted(os.listdir(frame_dir))
        if len(frame_files) == 0:
            raise FileNotFoundError(frame_dir)
        frame_idx = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
        video = [
            Image.open(os.path.join(frame_dir, frame_files[i])).convert("RGB")
            for i in frame_idx
        ]

        video_processed = []
        for frame in video:
            processed_frame = process_anyres_video(frame, self.image_processor)
            video_processed.append(processed_frame)

        video_final = torch.cat(video_processed, dim=0).bfloat16().to("cuda").unsqueeze(0)
        video_data = ((video_final, video_final), (384, 384), "video")
        return video_data
     

    def get_feature_IT(self, data_packet, model="train"):
        T_class_hidden = []
        v_class_hidden = []
        for step,sid in enumerate(data_packet[-1]):
            text_raw = data_packet[1][step]
            image_file = data_packet[0][step]
            image = Image.open(image_file).convert('RGB')
            # 1. Speech_input
            speechs = [torch.zeros(1, 3000, 128).bfloat16().to('cuda')]
            speech_lengths = [torch.LongTensor([3000]).to('cuda')]
            speech_wavs = [torch.zeros([1, 480000]).to('cuda')]
            speech_chunks = [torch.LongTensor([1]).to('cuda')]
            # 2. Text_input
            conv_mode = "qwen_1_5"
            qs = text_raw
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], "[CLASS]")
            prompt = conv.get_prompt()
            input_ids = tokenizer_speech_token(prompt, self.tokenizer, SPEECH_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')

            attention_masks = input_ids.ne(self.pad_token_ids).long().to('cuda')
            with torch.inference_mode():
                T_hidden = self.model(input_ids=input_ids, 
                                      images=[torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16, device='cuda', non_blocking=True)], 
                                      images_highres=[torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16, device='cuda', non_blocking=True)],
                                      image_sizes=[(224, 224)], 
                                      modalities=['text'], 
                                      speech=speechs, 
                                      speech_lengths=speech_lengths, 
                                      speech_chunks=speech_chunks, 
                                      speech_wav=speech_wavs, attention_mask=attention_masks, 
                                      output_hidden_states=True, use_cache=True)
                T_tmp_hidden = T_hidden["hidden_states"][-1][0][-3]
                T_class_hidden.append(T_tmp_hidden)
            
            # 3. Image input

            conv_mode = "qwen_1_5"
            qs = DEFAULT_IMAGE_TOKEN
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], "[CLASS]")
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
            self.image_processor.do_resize = False
            self.image_processor.do_center_crop = False
            image_tensor, image_highres_tensor = process_anyres_highres_image(image, self.image_processor)
            image_tensor, image_highres_tensor = image_tensor.bfloat16().to("cuda").unsqueeze(0), image_highres_tensor.bfloat16().to("cuda").unsqueeze(0)
            image_size = [image.size]

            attention_masks = input_ids.ne(self.pad_token_ids).long().to('cuda')
            with torch.inference_mode():
                v_hidden = self.model(input_ids=input_ids, 
                                      images=image_tensor, 
                                      images_highres=image_highres_tensor, 
                                      modalities="image", speech=speechs, 
                                      speech_lengths=speech_lengths, speech_chunks=speech_chunks, 
                                      speech_wav=speech_wavs, attention_mask=attention_masks, 
                                      output_hidden_states=True, use_cache=True)
                v_tmp_hidden = v_hidden["hidden_states"][-1][0][-3]
                v_class_hidden.append(v_tmp_hidden)
            
            one_path = os.path.join(self.OLA_dir, f"{sid}.pt")
            # print(f"a_class_hidden shape is {v_tmp_hidden.shape}")
            torch.save({
                "file_name": sid,
                "text": T_tmp_hidden,
                "image": v_tmp_hidden,
            }, one_path)
            
        T_class_hidden = torch.stack(T_class_hidden)
        v_class_hidden = torch.stack(v_class_hidden)
        return T_class_hidden,v_class_hidden
    
    def get_features_av(self, av_files, mode="train"):
        
        a_class_hidden = []
        v_class_hidden = []
        for file  in av_files:
            # start = time.time()
            audio_file = os.path.join(self.audio_feature_path, file + ".wav")
            speech, speech_length, speech_chunk, speech_wav = self.load_audio(audio_file)
            speechs = []
            speech_lengths = []
            speech_wavs = []
            speech_chunks = []
            

            
            speechs.append(speech.bfloat16().to('cuda'))
            speech_lengths.append(speech_length.to('cuda'))
            speech_chunks.append(speech_chunk.to('cuda'))
            speech_wavs.append(speech_wav.to('cuda'))

            images = [torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16, device='cuda', non_blocking=True)]
            images_highres = [torch.zeros(1, 3, 224, 224).to(dtype=torch.bfloat16, device='cuda', non_blocking=True)]
            image_sizes = [(224, 224)]
            

            conv_mode = "qwen_1_5"
            qs = DEFAULT_SPEECH_TOKEN + "\n" + ''
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], "[CLASS]")
            prompt = conv.get_prompt()
            input_ids = tokenizer_speech_token(prompt, self.tokenizer, SPEECH_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')

            attention_masks = input_ids.ne(self.pad_token_ids).long().to('cuda')

            # print(input_ids)
            # with torch.inference_mode():
            with torch.inference_mode():
                a_hidden = self.model(input_ids=input_ids, attention_mask=attention_masks,
                                      images=images, images_highres=images_highres, image_sizes=image_sizes, 
                                      speech=speechs, speech_lengths=speech_lengths, speech_chunks=speech_chunks, speech_wav=speech_wavs, 
                                      modalities=['text'],output_hidden_states=True, use_cache=True)
                a_tmp_hidden = a_hidden["hidden_states"][-1][0][-3]
                a_class_hidden.append(a_tmp_hidden)
            

            speechs = [torch.zeros(1, 3000, 128).bfloat16().to('cuda')]
            speech_lengths = [torch.LongTensor([3000]).to('cuda')]
            speech_wavs = [torch.zeros([1, 480000]).to('cuda')]
            speech_chunks = [torch.LongTensor([1]).to('cuda')]
            conv_mode = "qwen_1_5"
            qs = DEFAULT_IMAGE_TOKEN
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], "[CLASS]")
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to('cuda')
            self.image_processor.do_resize = False
            self.image_processor.do_center_crop = False
            if self.use_frame_dir:
                frame_dir = os.path.join(self.frame_feature_path, file)
                video_data = self.process_frames(frame_dir)
            else:
                visual_file = os.path.join(self.visual_feature_path, file + ".flv")
                video_data = self.process_video(visual_file)
            image_tensor, image_highres_tensor = video_data[0][0], video_data[0][1]

            attention_masks = input_ids.ne(self.pad_token_ids).long().to('cuda')
            with torch.inference_mode():
                v_hidden = self.model(input_ids=input_ids, images=image_tensor, 
                                      images_highres=image_highres_tensor, 
                                      modalities="video", speech=speechs, 
                                      speech_lengths=speech_lengths, 
                                      speech_chunks=speech_chunks, speech_wav=speech_wavs, 
                                      attention_mask=attention_masks, output_hidden_states=True, 
                                      use_cache=True)
                v_tmp_hidden = v_hidden["hidden_states"][-1][0][-3]
                v_class_hidden.append(v_hidden["hidden_states"][-1][0][-3])
            
            one_path = os.path.join(self.OLA_dir, f"{file}.pt")
            # print(f"a_class_hidden shape is {v_tmp_hidden.shape}")
            torch.save({
                "file_name": file,
                "audio": a_tmp_hidden,
                "video": v_tmp_hidden,
            }, one_path)
            
        a_class_hidden = torch.stack(a_class_hidden)
        v_class_hidden = torch.stack(v_class_hidden)

        
        return v_class_hidden, a_class_hidden


    def forward(self,
                datas,
                labels=None,
                epoch=-1,
                epoch_index=-1,
                drop=None,
                modality_idx=None,
                sid=None,
                mode="train",
               ):
        av_files = sid
        m1_feature, m2_feature = self.get_features_av(av_files, mode)
        outputs = self.fusion_model(m1_feature.float(), 
                                    m2_feature.float(),
                                    epoch=epoch,
                                    epoch_index=epoch_index,
                                    labels=labels,
                                    sid=sid,
                                   )
        return outputs
        
