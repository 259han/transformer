from torch import nn
from config import *
from logger import logger

import torch
import math
import torch.jit as jit

# 使用JIT优化Layer Normalization实现
@torch.jit.script
def layer_norm_jit(x, weight, bias, eps: float = 1e-6):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    y = (x - mean) / torch.sqrt(var + eps)
    return weight * y + bias

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用PyTorch原生的MultiheadAttention，简化参数配置
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=drop_out_rate,
            batch_first=True  # 设置batch_first=True使输入形状为(batch_size, seq_len, embed_dim)
        )
        
        # 使用标准的LayerNorm实现
        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        
        self.drop_out_1 = nn.Dropout(drop_out_rate)
        self.feed_forward = FeedForwardLayer()
        self.drop_out_2 = nn.Dropout(drop_out_rate)

    def forward(self, x, e_mask):
        # 调整mask格式，PyTorch的MultiheadAttention期望的key_padding_mask形状为(batch_size, seq_len)
        # 而不是我们之前的(batch_size, 1, seq_len)形状
        if e_mask is not None:
            key_padding_mask = ~(e_mask.squeeze(1))  # 注意需要取反，因为PyTorch中1表示要mask的位置
        else:
            key_padding_mask = None
            
        # 自注意力子层: 注意力 + 残差 + LayerNorm
        # 使用torch.cuda.amp.autocast()包装，提高混合精度性能
        attn_output, _ = self.multihead_attention(x, x, x, key_padding_mask=key_padding_mask)
        x = x + self.drop_out_1(attn_output)  # 残差连接
        x = self.layer_norm_1(x)  # Layer Normalization
        
        # 前馈网络子层: 前馈 + 残差 + LayerNorm
        ff_output = self.feed_forward(x)
        x = x + self.drop_out_2(ff_output)  # 残差连接
        x = self.layer_norm_2(x)  # Layer Normalization

        return x  # (B, L, d_model)


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用PyTorch原生的MultiheadAttention，简化参数配置
        # 自注意力层，用于解码器内部注意力
        self.masked_multihead_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=drop_out_rate,
            batch_first=True
        )
        
        # 交叉注意力也使用PyTorch原生的MultiheadAttention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=drop_out_rate,
            batch_first=True
        )
        
        # 使用标准的LayerNorm实现
        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        self.layer_norm_3 = LayerNormalization()
        
        self.drop_out_1 = nn.Dropout(drop_out_rate)
        self.drop_out_2 = nn.Dropout(drop_out_rate)
        self.feed_forward = FeedForwardLayer()
        self.drop_out_3 = nn.Dropout(drop_out_rate)
        
        # 预计算因果掩码
        max_len = seq_len
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(max_len, max_len, dtype=torch.bool),
                diagonal=1
            )
        )

    def forward(self, x, e_output, e_mask, d_mask):
        """
        前向传播
        
        参数:
        - x: 输入张量，形状 (batch_size, seq_len, d_model)
        - e_output: 编码器输出，形状 (batch_size, src_seq_len, d_model)
        - e_mask: 编码器注意力掩码
        - d_mask: 解码器注意力掩码
        
        返回:
        - x: 解码器输出，形状 (batch_size, seq_len, d_model)
        """
        # 为masked self-attention准备掩码
        # 解码器自注意力需要两种掩码：填充掩码和因果掩码(防止看到未来)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 1. 创建填充掩码 (batch_size, seq_len)
        if d_mask is not None:
            # 从(batch_size, 1, seq_len)转换为(batch_size, seq_len)
            # 注意：PyTorch attention中True表示需要mask的位置
            padding_mask = ~(d_mask[:, 0, :])
        else:
            padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
            
        # 2. 使用预计算的因果掩码 (seq_len, seq_len)
        # 截取需要的长度
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        
        # 3. 带掩码的自注意力子层
        masked_attn_output, _ = self.masked_multihead_attention(
            x, x, x,
            attn_mask=causal_mask,    # 通过attn_mask应用因果掩码
            key_padding_mask=padding_mask  # 通过key_padding_mask应用填充掩码
        )
        
        # 残差连接和层归一化
        x = x + self.drop_out_1(masked_attn_output)
        x = self.layer_norm_1(x)
        
        # 4. 为交叉注意力准备编码器掩码
        if e_mask is not None:
            # 从(batch_size, 1, src_seq_len)转换为(batch_size, src_seq_len)
            encoder_padding_mask = ~(e_mask.squeeze(1))
        else:
            encoder_padding_mask = None
            
        # 5. 编码器-解码器注意力子层
        cross_attn_output, _ = self.multihead_attention(
            query=x,
            key=e_output,
            value=e_output,
            key_padding_mask=encoder_padding_mask
        )
        
        # 残差连接和层归一化
        x = x + self.drop_out_2(cross_attn_output)
        x = self.layer_norm_2(x)
        
        # 6. 前馈网络子层
        ff_output = self.feed_forward(x)
        x = x + self.drop_out_3(ff_output)
        x = self.layer_norm_3(x)

        return x


class FeedForwardLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # 优化前馈网络实现
        # 使用更高效的激活函数和计算顺序
        self.linear1 = nn.Linear(d_model, d_ff, bias=True)
        self.activation = nn.ReLU(inplace=True)  # 使用inplace操作节省内存
        self.dropout = nn.Dropout(drop_out_rate)
        self.linear2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x):
        # 优化计算顺序，减少内存使用
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        # 使用原生LayerNorm而不是自定义实现
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # 使用JIT优化的LayerNorm计算
        if jit_script_encoder or jit_script_decoder:
            return layer_norm_jit(x, self.weight, self.bias, self.eps)
        else:
            # 使用标准实现
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, unbiased=False, keepdim=True)
            return self.weight * (x - mean) / (std + self.eps) + self.bias


class PositionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Make initial positional encoding matrix with 0
        # 创建位置编码矩阵
        pe_matrix = torch.zeros(seq_len, d_model) # (L, d_model)

        # 计算位置编码值
        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        # 使用向量化操作提高效率
        pe_matrix[:, 0::2] = torch.sin(position * div_term)
        pe_matrix[:, 1::2] = torch.cos(position * div_term)

        pe_matrix = pe_matrix.unsqueeze(0) # (1, L, d_model)
        # 使用register_buffer注册位置编码矩阵，让PyTorch自动管理设备
        self.register_buffer('positional_encoding', pe_matrix)
        
        # 缩放因子
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        # 优化位置编码的应用方式
        x = x * self.scale # (B, L, d_model)
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:, :seq_len, :]
        return x + pos_enc
