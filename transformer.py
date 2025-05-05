import torch
from torch import nn
from config import *
from layers import *
from logger import logger

import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class Transformer(nn.Module):
    """
    Transformer模型实现
    基于论文 "Attention is All You Need" 
    """
    def __init__(self, src_vocab_size, trg_vocab_size):
        """
        初始化Transformer模型
        
        参数:
        - src_vocab_size: 源语言词汇表大小
        - trg_vocab_size: 目标语言词汇表大小
        """
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        
        logger.info(f"初始化Transformer模型: d_model={d_model}, num_heads={num_heads}, num_layers={num_layers}")
        logger.info(f"源语言词汇表大小: {src_vocab_size}, 目标语言词汇表大小: {trg_vocab_size}")

        # 创建词嵌入
        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, d_model)
        
        # 位置编码
        self.positional_encoder = PositionalEncoder()
        
        # 编码器和解码器
        self.encoder = Encoder()
        self.decoder = Decoder()
        
        # 输出层
        self.output_linear = nn.Linear(d_model, self.trg_vocab_size)
        
        # 初始化模型参数
        self._init_parameters()
        
        # 记录是否使用梯度检查点
        self.use_checkpointing = False
        
        logger.info("Transformer模型初始化完成")

    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # 用更适中的增益值初始化输出层
        nn.init.xavier_uniform_(self.output_linear.weight, gain=0.5)
        
        logger.debug("模型参数已使用Xavier均匀分布初始化")
        
    def enable_checkpointing(self, use_checkpointing=True):
        """启用或禁用梯度检查点以节省内存"""
        self.use_checkpointing = use_checkpointing
        logger.info(f"梯度检查点: {'启用' if use_checkpointing else '禁用'}")
        return self

    def forward(self, src_input, trg_input, e_mask=None, d_mask=None):
        """
        前向传播
        
        参数:
        - src_input: 源语言输入，形状 (batch_size, src_seq_len)
        - trg_input: 目标语言输入，形状 (batch_size, trg_seq_len)
        - e_mask: 编码器注意力掩码，用于屏蔽填充部分
        - d_mask: 解码器注意力掩码，用于屏蔽填充部分和未来位置
        
        返回:
        - output: 模型输出，形状 (batch_size, trg_seq_len, trg_vocab_size)
        """
        # 词嵌入和位置编码
        src_embedded = self.positional_encoder(self.src_embedding(src_input))
        trg_embedded = self.positional_encoder(self.trg_embedding(trg_input))

        # 编码器处理
        if self.use_checkpointing and self.training:
            # 使用梯度检查点节省内存
            e_output = checkpoint.checkpoint(self.encoder, src_embedded, e_mask)
        else:
            e_output = self.encoder(src_embedded, e_mask) # (B, L, d_model)
        
        # 解码器处理
        if self.use_checkpointing and self.training:
            # 使用梯度检查点节省内存
            d_output = checkpoint.checkpoint(
                self.decoder, trg_embedded, e_output, e_mask, d_mask
            )
        else:
            d_output = self.decoder(trg_embedded, e_output, e_mask, d_mask) # (B, L, d_model)

        # 输出层处理
        return self.output_linear(d_output)


class Encoder(nn.Module):
    """Transformer编码器，包含多个编码器层"""
    def __init__(self):
        """初始化编码器"""
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(num_layers)])
        self.layer_norm = LayerNormalization()
            
        logger.debug(f"编码器初始化完成，层数: {num_layers}")

    def forward(self, x, e_mask):
        """
        前向传播
        
        参数:
        - x: 输入张量，形状 (batch_size, seq_len, d_model)
        - e_mask: 编码器注意力掩码
        
        返回:
        - x: 编码器输出，形状 (batch_size, seq_len, d_model)
        """
        # 依次通过每个编码器层
        for layer in self.layers:
            x = layer(x, e_mask)
        
        # 最后做一次层归一化
        return self.layer_norm(x)


class Decoder(nn.Module):
    """Transformer解码器，包含多个解码器层"""
    def __init__(self):
        """初始化解码器"""
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(num_layers)])
        self.layer_norm = LayerNormalization()
            
        logger.debug(f"解码器初始化完成，层数: {num_layers}")

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
        # 依次通过每个解码器层
        for layer in self.layers:
            x = layer(x, e_output, e_mask, d_mask)
        
        # 最后做一次层归一化
        return self.layer_norm(x)
