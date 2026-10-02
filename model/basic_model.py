import torch
import torch.nn as nn
import pickle as pickle
import torch.nn.functional as F

from transformers import BertModel

from .AME import AME
from . import mae as MAE
from .utils import weight_init
from .backbone import resnet18
from .fusion_model import ConcatFusion,SumFusion,GatedFusion,FiLM
from .fusion_model import ConcatTVA,SumTVA,GatedTVA,FiLMTVA
from .fusion_model import ConcatFusion3_MLA, ConcatFusion_MLA
from .fusion_model import CAfusion, MMTMFusion, CentralNetFusion

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
# text encoder, based on BERT
class TextEncoder(nn.Module):
    """
    Encodes pre-tokenized text using a pretrained BERT model from Hugging Face.
    The forward method expects input_ids and attention_mask tensors.
    """
    def __init__(self,
                 args,
                 model_path="/root/autodl-tmp/AMRe/model/bert-base-uncased",
                ):
        super(TextEncoder, self).__init__()
        # Load the pretrained BERT model.
        self.model = BertModel.from_pretrained(model_path)
        
    def forward(self, input_ids, attention_mask): 
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Return the last hidden state (all token embeddings).
        text_emb = outputs.last_hidden_state
        return text_emb
        # # Use the [CLS] embedding (first token).
        # cls_emb = outputs.last_hidden_state[:, 0, :]
        # return cls_emb

# visual encoder, based on MAE
class MAE_VideoEncoder(nn.Module):
    def __init__(self, args):
        super(MAE_VideoEncoder, self).__init__()
        self.MAE_model = MAE.__dict__["mae_vit_base_patch16"](norm_pix_loss=False)
            
        checkpoint = torch.load(
            "/root/autodl-tmp/AMRe/model/mae_pretrain_vit_base.pth", map_location="cpu"
        )
        self.MAE_model.load_state_dict(checkpoint["model"], strict=False)

    def forward(self, image):
        image_features = self.MAE_model.forward_ying(image)
        return image_features

# video encoder, based on ResNet18
class _ResNet18_V(nn.Module):
    def __init__(self, args, output_dim=512):
        super(_ResNet18_V, self).__init__()
        self.basic_resnet18 = resnet18(modality="visual", args=args)
        if output_dim != 512:
            # to compatible with text model output dim (768)
            self.out_conv = nn.Conv2d(512 ,
                                      output_dim, kernel_size=1, stride=1, bias=False)
        
    def forward(self, x):
        # first, we reshape the input tensor (B, C, T, H, W) -> (B*T, C, H, W)
        # which means we stack the frames of the video
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)

        x = self.basic_resnet18(x)

        # compatible with text model output dim
        if hasattr(self, 'out_conv'):
            x = self.out_conv(x)
        
        # recover the original shape
        (_, C, H, W) = x.size()
        x = x.view(B, -1, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # average pooling over the frames
        out = F.adaptive_avg_pool3d(x, 1)
        
        # flatten the tensor
        out = torch.flatten(out, 1)
        return out
    
# audio encoder, based on ResNet18
class _ResNet18_A(nn.Module):
    def __init__(self, args, output_dim=512):
        super(_ResNet18_A, self).__init__()
        self.basic_resnet18 = resnet18(modality="audio", args=args)
        
        if output_dim != 512:
            # to compatible with text model output dim
            self.out_conv = nn.Conv2d(512,
                                      output_dim, kernel_size=1, stride=1, bias=False)
        
    def forward(self, audio_spectrogram):
        out = self.basic_resnet18(audio_spectrogram)
        # 只有当 out_conv 存在时（即 output_dim != 512 时）才通过 1x1 卷积改变通道数，避免在默认情况下额外计算
        if hasattr(self, 'out_conv'):
            out = self.out_conv(out)
            
        out = F.adaptive_avg_pool2d(out, 1)

        out = torch.flatten(out, 1)
        return out

# For testing single modality
# Single modality (visual or audio) model, based on ResNet18
class UniR18(nn.Module):
    def __init__(self, args, feature_dim=512):
        super(UniR18, self).__init__()
        
        self.modality = args.modality

        n_classes = args.num_classes

        if args.modality == 'audio':
            self.net = _ResNet18_A(feature_dim)

        elif args.modality == 'visual':
            self.net = _ResNet18_V(feature_dim)
        else:
            raise NotImplementedError(
                'Incorrect modality: {}!'.format(args.modality))

        self.fc = nn.Linear(feature_dim, n_classes)
        self.feature = None
        
    def forward(self, x):
        self.feature = self.net(x)
        m = self.fc(self.feature)
        return m

    def get_feature_embedding(self):
        return self.feature

# Single modality (text) model, based on BERT
class UniBERT(nn.Module):
    def __init__(self, args):
        super(UniBERT, self).__init__()
        
        self.modality = args.modality

        n_classes = args.num_classes

        self.text_encoder = TextEncoder(model_path="bert-base-uncased", fine_tune=True, unfreeze_last_n_layers=5)
        
        print("BERT model loaded")
        print("n_classes: ", n_classes)
        print("hidden size: ", self.text_encoder.model.config.hidden_size)
        
        self.fc = nn.Linear(self.text_encoder.model.config.hidden_size, n_classes)
        self.feature = None
        
    def forward(self, input_ids, attention_mask):
        # Get the full hidden states from text encoder
        hidden_states = self.text_encoder(input_ids, attention_mask)
        # Use [CLS] token embedding (first token) for classification
        cls_embedding = hidden_states[:, 0, :]
        self.feature = cls_embedding
        m = self.fc(cls_embedding)
        return m

    def get_feature_embedding(self):
        # size should be (batch_size, 768) for BERT-base
        return self.feature


class VA_Classifier(nn.Module):
    def __init__(self, args, feature_dim=512):
        super(VA_Classifier, self).__init__()
        self.args = args
        if args.model_name == '["Visual","Audio"]':
            self.encoder_1 = _ResNet18_V(args, feature_dim) # visual encoder
            self.encoder_2 = _ResNet18_A(args, feature_dim) # audio encoder
            # # weight init
            # self.encoder_1.apply(weight_init)
            # self.encoder_2.apply(weight_init)
            
        elif args.model_name == '["Image","Text"]':
            self.encoder_1 = MAE_VideoEncoder(args) # image encoder
            # self.encoder_1 = _ResNet18_V(args,output_dim=args.unified_dim)
            # self.encoder_1.apply(weight_init)
            self.encoder_2 = TextEncoder(args=args, model_path="/root/autodl-tmp/AMRe/model/bert-base-uncased") # text encoder
        fusion = args.fusion_method
        if fusion == "concat":
            print(f"***Use Concat Fusion Model***")
            self.fusion_model = ConcatFusion(args)
        elif fusion == 'sum':
            print(f"***Use Sum Fusion Model***")
            self.fusion_model = SumFusion(args)
        elif fusion == 'Gate':
            print(f"***Use Gated Fusion Model***")
            self.fusion_model = GatedFusion(args)
        elif fusion == 'Film':
            print(f"***Use FiLM Fusion Model***")
            self.fusion_model = FiLM(args)
        elif fusion == "CA": # 表示有参数融合
            self.fusion_model = CAfusion(args)
        elif fusion == "MMTM":
            self.fusion_model = MMTMFusion(args)
        elif fusion == "CentralNet":
            self.fusion_model = CentralNetFusion(args)

    def get_Mask(self, datas, epoch=-1, epoch_index=-1, labels=None):
        if self.args.model_name == '["Visual","Audio"]':
            video,audio_spectrogram=datas
            if video is None:
                # m1_feature = self.encoder_1(video)
                m2_feature = self.encoder_2(audio_spectrogram)
                m1_feature = torch.zeros_like(m2_feature)
            elif audio_spectrogram is None:
                m1_feature = self.encoder_1(video)
                m2_feature = torch.zeros_like(m1_feature)
            else:
                m1_feature = self.encoder_1(video)
                m2_feature = self.encoder_2(audio_spectrogram)
            # if self.args.model_name == '["Visual","Audio"]':
            m1_feature = m1_feature.unsqueeze(1)  # (B, 1, D)
            m2_feature = m2_feature.unsqueeze(1)  # (B, 1, D)
            m1_feature = m1_feature.mean(dim=1)  # 全局平均池化
            m2_feature = m2_feature.mean(dim=1)  # 全局平均池化
        elif self.args.model_name == '["Image","Text"]':
            token,padding_mask,image = datas
            # image = image.unsqueeze(2)
            # print(f"image shape is {image.shape}")
            m1_feature = self.encoder_1(image)
            
            # 文本编码
            token = token.squeeze(1)
            padding_mask = padding_mask.squeeze(1)
            # print(f"padding mask shape is {padding_mask.shape}")
            m2_feature = self.encoder_2(input_ids=token,attention_mask=padding_mask)
            
            m1_feature = m1_feature[:,0,:]  # 取CLS token
            m2_feature = m2_feature[:,0,:]  # 取CLS token
        
        device = m1_feature.device
        batch_size = m1_feature.shape[0]
        # epoch=-1
        m1_feature, m2_feature, m1_mask, m2_mask = self.fusion_model.AME_MASK(epoch,batch_size,m1_feature,m2_feature,device,labels=labels,epoch_index=epoch_index)

        return m1_mask,m2_mask
    
        
    def forward(self, 
                datas,
                labels=None,
                epoch=-1,
                epoch_index=-1,
                drop=None,
                modality_idx=None,
                sid=None,
               ):
        if self.args.model_name == '["Visual","Audio"]':
            video,audio_spectrogram=datas
            if video is None:
                # m1_feature = self.encoder_1(video)
                m2_feature = self.encoder_2(audio_spectrogram)
                m1_feature = torch.zeros_like(m2_feature)
            elif audio_spectrogram is None:
                m1_feature = self.encoder_1(video)
                m2_feature = torch.zeros_like(m1_feature)
            else:
                m1_feature = self.encoder_1(video)
                m2_feature = self.encoder_2(audio_spectrogram)
            # if self.args.model_name == '["Visual","Audio"]':
            m1_feature = m1_feature.unsqueeze(1)  # (B, 1, D)
            m2_feature = m2_feature.unsqueeze(1)  # (B, 1, D)
        elif self.args.model_name == '["Image","Text"]':
            token,padding_mask,image = datas
            # image = image.unsqueeze(2)
            # print(f"image shape is {image.shape}")
            if image is None:
                # 文本编码
                token = token.squeeze(1)
                padding_mask = padding_mask.squeeze(1)
                # print(f"padding mask shape is {padding_mask.shape}")
                m2_feature = self.encoder_2(input_ids=token,attention_mask=padding_mask)
                m1_feature = torch.zeros_like(m2_feature)
            elif token is None:
                
                m1_feature = self.encoder_1(image)
                m2_feature = torch.zeros_like(m1_feature)
            else:
                m1_feature = self.encoder_1(image)
                # 文本编码
                token = token.squeeze(1)
                padding_mask = padding_mask.squeeze(1)
                # print(f"padding mask shape is {padding_mask.shape}")
                m2_feature = self.encoder_2(input_ids=token,attention_mask=padding_mask)
            
            if m1_feature.dim() == 2:
                m1_feature = m1_feature.unsqueeze(1)
            # print(f"m1_feature shape is {m1_feature.shape}")
            # print(f"m2_feature shape is {m2_feature.shape}")
        # m2_feature = torch.zeros_like(m1_feature,device=m1_feature.device)
        # visual/Image:2, audio/Text: 1
        if drop != None:
            # print(drop)
            for i in range(len(drop)):
                # if drop[i] == 1:
                #     m1_feature[i,:] = 0.0
                # elif drop[i] == 2:
                #     m2_feature[i,:] = 0.0
                if drop[i] == 2:
                    m1_feature[i,:] = 0.0
                elif drop[i] == 1:
                    m2_feature[i,:] = 0.0
        if modality_idx == 1:
            m2_feature = torch.zeros_like(m2_feature)
        elif modality_idx == 2:
            m1_feature = torch.zeros_like(m1_feature)
        # print(f"modality_idx is {modality_idx}")
        # print(f"m1_feature is {m1_feature}")
        # print(f"m2_feature is {m2_feature}")
        outputs = self.fusion_model(m1_feature, 
                                   m2_feature,
                                   epoch=epoch,
                                   epoch_index=epoch_index,
                                   labels=labels,
                                   sid=sid,
                                  )
        drop_info = getattr(self.fusion_model, 'latest_drop_info', None)
        if drop_info is not None:
            outputs = (*outputs, drop_info)
        return outputs
        # return fusion_out,out_1,out_2,m1_mask,m2_mask


class TVA_Classifier(nn.Module):
    def __init__(self, args, output_dim=768):
        super(TVA_Classifier, self).__init__()
        self.args = args
        if args.model_name == '["Text","Visual","Audio"]':
            self.encoder_1 = TextEncoder(args=args, model_path="/root/autodl-tmp/AMRe/model/bert-base-uncased") # text encoder
            self.encoder_2 = _ResNet18_V(args, output_dim=output_dim) # visual encoder
            # self.encoder_2 = MAE_VideoEncoder(args) # image encoder
            self.encoder_3 = _ResNet18_A(args, output_dim=output_dim) # audio encoder
            # # weight init
            # self.encoder_1.apply(weight_init)
            # self.encoder_2.apply(weight_init)
            # self.encoder_3.apply(weight_init)
    
        fusion = args.fusion_method
        if fusion == "concat":
            self.fusion_model = ConcatTVA(args)
        elif fusion == "sum": # 表示有参数融合
            self.fusion_model = SumTVA(args)
        elif fusion == "Gate":
            self.fusion_model = GatedTVA(args)
        elif fusion == "Film":
            self.fusion_model = FiLMTVA(args)
        elif fusion == "MLA":
            pass
        elif fusion == "SUM":
            pass

    def get_Mask(self, datas, epoch=-1, epoch_index=-1, labels=None):
        token,padding_mask,image,audio_spectrogram=datas
        m1_feature = self.encoder_1(input_ids=token,attention_mask=padding_mask)
        m2_feature = self.encoder_2(image)
        m3_feature = self.encoder_3(audio_spectrogram)
        m2_feature = m2_feature.unsqueeze(1)  # (B, 1, D)
        m3_feature = m3_feature.unsqueeze(1)  # (B, 1, D)
        
        m1_feature = m1_feature.mean(dim=1)  # 全局平均池化
        m2_feature = m2_feature.mean(dim=1)  # 全局平均池化
        m3_feature = m3_feature.mean(dim=1)  # 全局平均池化

        # epoch=-1
        device = m1_feature.device
        batch_size = m1_feature.shape[0]
        _, _, _, m1_mask, m2_mask, m3_mask = self.fusion_model.AME_MASK(epoch,batch_size,m1_feature,m2_feature, m3_feature,device,labels=labels,epoch_index=-1)

        return m1_mask,m2_mask, m3_mask
    
    def forward(self, 
                datas,
                labels=None,
                epoch=-1,
                epoch_index=-1,
                drop=None,
                modality_idx=None,
                sid=None,
               ):
        token,padding_mask,image,audio_spectrogram=datas
        # print(f"audio_spectrogram shape is {audio_spectrogram.shape}")
        # print(f"iamge shape is {image.shape}")
        # if token is not None:
        # batch_size = image.size[1]
        if image is None and audio_spectrogram is None:
            m1_feature = self.encoder_1(input_ids=token,attention_mask=padding_mask)
            batch_size = token.shape[0]
            m2_feature = torch.zeros(batch_size,768,device='cuda:0')
            m3_feature = torch.zeros(batch_size,768,device='cuda:0')
        # elif image is not None:
        elif token is None and audio_spectrogram is None:
            m2_feature = self.encoder_2(image)
            m1_feature = torch.zeros_like(m2_feature).unsqueeze(1)
            m3_feature = torch.zeros_like(m2_feature)
        # elif audio_spectrogram is not None:
        elif token is None and image is None:
            m3_feature = self.encoder_3(audio_spectrogram)
            m1_feature = torch.zeros_like(m3_feature).unsqueeze(1)
            m2_feature = torch.zeros_like(m3_feature)
        elif token is None:
            m2_feature = self.encoder_2(image)
            m3_feature = self.encoder_3(audio_spectrogram)
            m1_feature = torch.zeros_like(m2_feature).unsqueeze(1)
        elif image is None:
            m1_feature = self.encoder_1(input_ids=token,attention_mask=padding_mask)
            # m2_feature = self.encoder_2(image)
            m3_feature = self.encoder_3(audio_spectrogram)
            m2_feature = torch.zeros_like(m3_feature)
        elif audio_spectrogram is None:
            m1_feature = self.encoder_1(input_ids=token,attention_mask=padding_mask)
            m2_feature = self.encoder_2(image)
            # m3_feature = self.encoder_3(audio_spectrogram)
            m3_feature = torch.zeros_like(m2_feature)
        else:
            m1_feature = self.encoder_1(input_ids=token,attention_mask=padding_mask)
            m2_feature = self.encoder_2(image)
            m3_feature = self.encoder_3(audio_spectrogram)
        
        m1_feature = m1_feature               # (B, 128, D)
        m2_feature = m2_feature.unsqueeze(1)  # (B, 1, D)
        # m2_feature = m2_feature               # (B, 128, D)
        m3_feature = m3_feature.unsqueeze(1)  # (B, 1, D)
        if drop != None:
            for i in range(len(drop)):
                if drop[i] == 1:
                    # print(f"m1_feature is 0")
                    m3_feature[i,:] = 0.0  # 丢弃text
                elif drop[i] == 2:
                    m2_feature[i,:] = 0.0  # 丢弃visual
                elif drop[i] == 3:
                    m1_feature[i,:] = 0.0  # 丢弃audio
        if modality_idx == 1:
            m2_feature = torch.zeros_like(m2_feature)
            m3_feature = torch.zeros_like(m3_feature)
        elif modality_idx == 2:
            m1_feature = torch.zeros_like(m1_feature)
            m3_feature = torch.zeros_like(m3_feature)
        elif modality_idx == 3:
            m1_feature = torch.zeros_like(m1_feature)
            m2_feature = torch.zeros_like(m2_feature)
        outputs = self.fusion_model(m1_feature, 
                                    m2_feature,
                                    m3_feature,
                                    epoch=epoch,
                                    epoch_index=epoch_index,
                                    labels=labels,
                                    sid=sid,
                                   )
        drop_info = getattr(self.fusion_model, 'latest_drop_info', None)
        if drop_info is not None:
            outputs = (*outputs, drop_info)
        return outputs
        # fusion_out,out_1,out_2,out_3,m1_mask,m2_mask,m3_mask = self.fusion_model(m1_feature, 
        #                                                                          m2_feature,
        #                                                                          m3_feature,
        #                                                                          epoch=epoch,
        #                                                                          epoch_index=epoch_index,
        #                                                                          labels=labels,
        #                                                                          sid=sid,
        #                                                                         )
        # return fusion_out,out_1,out_2,out_3,m1_mask,m2_mask,m3_mask


# MLA version
class VA_MLA_Classifier(nn.Module):
    def __init__(self, args):
        super(VA_MLA_Classifier, self).__init__()
        self.args = args
        feature_dim = args.unified_dim
        if args.model_name == '["Visual","Audio"]':
            self.encoder_1 = _ResNet18_V(args, feature_dim) # visual encoder
            self.encoder_2 = _ResNet18_A(args, feature_dim) # audio encoder
        elif args.model_name == '["Image","Text"]':
            self.encoder_1 = MAE_VideoEncoder(args) # image encoder
            self.encoder_2 = TextEncoder(args=args, model_path="/root/autodl-tmp/AMRe/model/bert-base-uncased") # text encoder
        self.fusion_model = ConcatFusion_MLA(input_dim=feature_dim, output_dim=args.num_classes)
    
    def forward(self, 
                datas,
                labels=None,
                epoch=-1,
                epoch_index=-1,
               ):
        if self.args.model_name == '["Visual","Audio"]':
            video,audio_spectrogram=datas
            m1_feature = self.encoder_1(video)
            m2_feature = self.encoder_2(audio_spectrogram)
            
        elif self.args.model_name == '["Image","Text"]':
            token,padding_mask,image = datas
            m1_feature = self.encoder_1(image)
            
            # 文本编码
            token = token.squeeze(1)
            padding_mask = padding_mask.squeeze(1)
            m2_feature = self.encoder_2(input_ids=token,attention_mask=padding_mask)
            m1_feature = m1_feature[:,0,:]
            m2_feature = m2_feature[:,0,:]  # 使用CLS向量

        return m1_feature,m2_feature
    
class TVA_MLA_Classifier(nn.Module):
    def __init__(self, args):
        super(TVA_MLA_Classifier, self).__init__()
        self.args = args
        output_dim = args.unified_dim
        if args.model_name == '["Text","Visual","Audio"]':
            self.encoder_1 = TextEncoder(args=args, model_path="/root/autodl-tmp/AMRe/model/bert-base-uncased") # text encoder
            self.encoder_2 = _ResNet18_V(args, output_dim=output_dim) # visual encoder
            self.encoder_3 = _ResNet18_A(args, output_dim=output_dim) # audio encoder
            # # weight init
            # self.encoder_1.apply(weight_init)
            self.encoder_2.apply(weight_init)
            self.encoder_3.apply(weight_init)
        self.fusion_model = ConcatFusion3_MLA(input_dim=output_dim, output_dim=args.num_classes)

        
    def forward(self, 
                datas,
                labels=None,
                epoch=-1,
                epoch_index=-1,
               ):
        token,padding_mask,image,audio_spectrogram=datas
        # print(f"audio_spectrogram shape is {audio_spectrogram.shape}")
        m1_feature = self.encoder_1(input_ids=token,attention_mask=padding_mask)
        m2_feature = self.encoder_2(image)
        m3_feature = self.encoder_3(audio_spectrogram)
        
        m1_feature = m1_feature[:,0,:]               # (B, 128, D)

        
        return m1_feature,m2_feature,m3_feature






























