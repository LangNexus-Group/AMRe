import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block


# Initialize weights for different layers
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# ==============================注意力机制================================
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.5, ffn_hidden_scale=4):
        super().__init__()
        assert model_dim % num_heads == 0, "model_dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.dim_per_head = model_dim // num_heads

        self.linear_q = nn.Linear(model_dim, model_dim)
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)
        self.linear_out = nn.Linear(model_dim, model_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        hidden_dim = int(model_dim * ffn_hidden_scale)
        self.fnn = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        residual = query  # 残差保留原始 query

        q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.dim_per_head)
        k = self.linear_k(key).view(batch_size, -1, self.num_heads, self.dim_per_head)
        v = self.linear_v(value).view(batch_size, -1, self.num_heads, self.dim_per_head)

        q = q.transpose(1, 2)  # [B, heads, Q_len, head_dim]
        k = k.transpose(1, 2)  # [B, heads, K_len, head_dim]
        v = v.transpose(1, 2)  # [B, heads, V_len, head_dim]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_per_head ** 0.5)

        if mask is not None:
            # 统一扩展成 [B, heads, Q_len, K_len]
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = self.softmax(scores)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, v)  # [B, heads, Q_len, head_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)
        context = self.linear_out(context)

        # 第一次残差 + LayerNorm
        context = self.norm1(residual + self.residual_dropout(context))

        # 前馈网络 + 第二次残差 + LayerNorm
        ffn_out = self.fnn(context)
        context = self.norm2(context + self.ffn_dropout(ffn_out))

        return context

# =============================模态间交叉注意力机制================================
class CoAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.5, name="IT"):
        super().__init__()
        self.name = name
        self.text_to_image = MultiHeadAttention(model_dim, num_heads, dropout)
        self.image_to_text = MultiHeadAttention(model_dim, num_heads, dropout)

    def forward(self, text, image, mask_text=None, mask_image=None):
        # 文本以图像为 Key/Value
        text_updated = self.text_to_image(text, image, image, mask_image)
        # 图像以文本为 Key/Value
        image_updated = self.image_to_text(image, text, text, mask_text)
        return text_updated, image_updated

# =============================带有内部模态融合共享的模块================================
class CoAttentionExpert(nn.Module):
    """
    带有内部模态融合共享的模块
    """

    def __init__(self, dim=512, num_heads=4, name="IT"):
        super().__init__()
        # self.MMSA = MMultiHeadAttention(model_dim=dim, num_heads=num_heads)
        self.block = Block(dim=dim, num_heads=num_heads)  # 共享
        self.name = name
        self.co_attn = CoAttention(model_dim=dim, num_heads=num_heads, name=name)

    def forward(self, image_feat, text_feat):
        """
        输入的是image和text，按照前面的部分工作展示，应该先注意力，在进行CoAttention
        输入： [batch_size, seq_len, model_dim]
        输出： [batch_size, seq_len, model_dim]
        """
        # 1、前共享
        # 先进行一次共享注意力
        # image_feat = self.MMSA(image_feat, image_feat, image_feat, modality="Image")
        # text_feat = self.MMSA(text_feat, text_feat, text_feat, modality="Text")
        image_feat_sa = self.block(image_feat)
        text_feat_sa = self.block(text_feat)

        # 2. 模态间交叉注意力 (Cross-Attention)
        # 使用自注意力处理后的特征进行交叉注意力计算
        text_feat_out, image_feat_out = self.co_attn(text_feat_sa, image_feat_sa)
        return image_feat, text_feat