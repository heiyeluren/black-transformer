# -*- encoding: utf-8 -*-

'''
## Heiyeluren Black Transformer ##

Heiyeluren Black Transformer

author: heiyeluren
date: 2023/7/17
site: github.com/heiyeluren

description:

black-transformer 是一个轻量级模拟Transformer模型实现的概要代码，用于了解整个Transformer工作机制

'''

import numpy as np
import torch
from torch import Tensor
from typing import Optional, Any, Union, Callable
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import time
from utils import *


# 多头注意力层
class MultiHeadedAttention(nn.Module):
    def __init__(self, 
                 num_heads: int,
                 d_model: int,
                 dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # 假设v_dim总是等于k_dim
        self.k_dim = d_model // num_heads
        self.num_heads = num_heads
        self.proj_weights = clones(
            nn.Linear(d_model, d_model), 4)  # W^Q, W^K, W^V, W^O
        self.attention_score = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Optional[Tensor] = None):
        """
        参数:
            query: shape (batch_size, seq_len, d_model)
            key: shape (batch_size, seq_len, d_model)
            value: shape (batch_size, seq_len, d_model)
            mask: shape (batch_size, seq_len, seq_len). 由于我们假设所有数据都使用相同的掩码，因此这里的形状也等于（1，seq_len，seq-len）

        返回:
            out: shape (batch_size, seq_len, d_model). 多头注意力层的输出
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        # 1) 应用W^Q、W^K、W^V生成新的查询、键、值
        query, key, value \
            = [proj_weight(x).view(batch_size, -1, self.num_heads, self.k_dim).transpose(1, 2)
                for proj_weight, x in zip(self.proj_weights, [query, key, value])]  # -1 equals to seq_len

        # 2) 计算注意力得分和out
        out, self.attention_score = attention(query, key, value, mask=mask,
                                              dropout=self.dropout)

        # 3) "Concat" 输出
        out = out.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.num_heads * self.k_dim)

        # 4) 应用W^O以获得最终输出
        out = self.proj_weights[-1](out)

        return out


# 注意力计算
def attention(query: Tensor,
              key: Tensor,
              value: Tensor,
              mask: Optional[Tensor] = None,
              dropout: float = 0.1):
    """
    定义如何计算注意力得分
    参数:
        query: shape (batch_size, num_heads, seq_len, k_dim)
        key: shape(batch_size, num_heads, seq_len, k_dim)
        value: shape(batch_size, num_heads, seq_len, v_dim)
        mask: shape (batch_size, num_heads, seq_len, seq_len). Since our assumption, here the shape is
              (1, 1, seq_len, seq_len)
    返回:
        out: shape (batch_size, v_dim). 注意力头的输出。注意力分数：形状（seq_len，seq_ln）。
    """
    k_dim = query.size(-1)

    # shape (seq_len ,seq_len)，row: token，col: token记号的注意力得分
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(k_dim)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e10)

    attention_score = F.softmax(scores, dim=-1)

    if dropout is not None:
        attention_score = dropout(attention_score)

    out = torch.matmul(attention_score, value)

    return out, attention_score  # shape: (seq_len, v_dim), (seq_len, seq_lem)


# if __name__ == '__main__':
#     d_model = 8
#     seq_len = 3
#     batch_size = 6
#     num_heads = 2
#     # mask = None
#     mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0).unsqueeze(0)

#     input = torch.rand(batch_size, seq_len, d_model)
#     multi_attn = MultiHeadedAttention(
#         num_heads=num_heads, d_model=d_model, dropout=0.1)
#     out = multi_attn(query=input, key=input, value=input, mask=mask)
#     print(out.shape)
#     print(out)
