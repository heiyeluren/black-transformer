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

import math
import torch
import torch.nn as nn


# 输入嵌入（Input Embeddings）层的构建
'''
这个层的作用是将 tokens 的整数列表编码成相应的向量集合，以便后续可以输入到神经网络中.
为了解决能够体现词与词之间的关系，使得意思相近的词有相近的表示结果，这种方法即 Word Embedding（词嵌入）。
最方便的途径是设计一个可学习的权重矩阵 W，将词向量与这个矩阵进行点乘，即得到新的表示结果。
假设 “爱” 和 “喜欢” 这两个词经过 one-hot 后分别表示为 10000 和 00001，权重矩阵设计如下：
[ w00, w01, w02
  w10, w11, w12
  w20, w21, w22
  w30, w31, w32
  w40, w41, w42 ]
那么两个词点乘后的结果分别是 [w00, w01, w02] 和 [w40, w41, w42]，在网络学习过程中（这两个词后面通常都是接主语，如“你”，“他”等，或者在翻译场景，
它们被翻译的目标意思也相近，它们要学习的目标一致或相近），权重矩阵的参数会不断进行更新，从而使得 [w00, w01, w02] 和 [w40, w41, w42] 的值越来越接近。
我们还把向量的维度从5维压缩到了3维。因此，word embedding 还可以起到降维的效果。
另一方面，其实，可以将这种方式看作是一个 lookup table：对于每个 word，进行 word embedding 就相当于一个lookup操作，在表中查出一个对应结果。
'''
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()

        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    

# 实现的是 Transformer 模型中的位置编码（Positional Encoding）
'''
word embedding，我们获得了词与词之间关系的表达形式，但是词在句子中的位置关系还无法体现。
由于 Transformer 是并行地处理句子中的所有词，因此需要加入词在句子中的位置信息，结合了这种方式的词嵌入就是 Position Embedding
预定义一个函数，通过函数计算出位置信息,大概公式如下：
\begin{gathered}
PE_{(pos,2i)}=\sin{(pos/10000^{2i/d})} \\
P E_{(p o s,2i+1)}=\operatorname{cos}\left(p o s_{\substack{i=1}{\mathrm{osc}}/\mathrm{1}{\mathrm{999}}\mathrm{2}i/d\right) 
\end{gathered}
Transformer 模型使用自注意力机制来处理序列数据，即在编码器和解码器中分别使用自注意力机制来学习输入数据的表示。
由于自注意力机制只对序列中的元素进行注意力权重的计算，它没有固定位置的概念，
因此需要为序列中的元素添加位置信息以帮助 Transformer 模型学习序列中元素的位置。
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)  
        pe = torch.zeros(max_len, d_model)  # max_len代表句子中最多有几个词
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  # d_model即公式中的d
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # 原向量加上计算出的位置信息才是最终的embedding
        return self.dropout(x)
    

# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()       
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         #pe.requires_grad = False
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]

