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

# Environment: python 3.10.x, pytorch 2.0.0+cu118, transformers 4.27.1, pytorch_pretrained_bert 0.6.2
#
# Models download from: https://huggingface.co/models
# Reference: https://huggingface.co/transformers/model_doc/bert.html
#            https://blog.csdn.net/weixin_42223207/article/details/119336324
#
# Transformers - Input Tokenize & Embedding
#


# import package
# import sys
import math
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, BasicTokenizer


# load pytorch_pretrained_bert model #
# from transformers import AutoTokenizer as TransAutoTokenizer
# from transformers import BertTokenizer as TransBertTokenizer
# from transformers import BertModel as TransBertModel
# from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
# from pytorch_pretrained import BertTokenizer
# from pytorch_pretrained_bert import BertModel as TorchBertModel
# from pytorch_pretrained_bert import BertTokenizer as TorchBertTokenizer

# load pretrained model #
# torch_tokenizer  = TorchBertTokenizer.from_pretrained(model_path)
# torch_bert_model = TorchBertModel.from_pretrained(model_path)
# trans_bert_tokenizer  = TransBertTokenizer.from_pretrained(PRETRAIN_MODEL_PATH, use_auth_token=True)
# trans_bert_model      = TransBertModel.from_pretrained(PRETRAIN_MODEL_PATH)



# define pretrained model path
MODEL_BERT_BASE_ZH          = "D:/Data/Models/pretrain_model/roc-bert-base-zh"
MODEL_BERT_BASE_CHINESE      = "bert-base-chinese"
MODEL_BERT_BASE              = "bert-base-cased"

'''
MODEL_BERT_BASE              = "bert-base-cased"
MODEL_BERT_BASE_CHINESE      = "bert-base-chinese"
MODEL_BERT_BASE_ZH           = "D:/Data/Models/pretrain_model/roc-bert-base-zh"
MODEL_BERT_BASE_UNCASE_VOCAB = "D:/Data/Models/pretrain_model/bert-base-uncased/bert-base-uncased-vocab.txt"

'''

import tiktoken
enc = tiktoken.get_encoding("gpt2")

# 字节对编码过程，我的输出是[31373, 995]
encoding_res = enc.encode("我的名字是Black")
print(encoding_res)

# 字节对解码过程，解码结果：hello world
raw_text = enc.decode(encoding_res)
print(raw_text)


# Tokenizer class
class Tokenizer:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
    
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)
    
    def convert_tokens_to_string(self, tokens):
        return self.tokenizer.convert_tokens_to_string(tokens)
    

# Input Sequence Embedding class
class InputEmbedding:
    def __init__(self, model_path):
        self.embedding_model = BertModel.from_pretrained(model_path)
        self.tokenizer       = BertTokenizer.from_pretrained(model_path)

    def get_seq_embedding(self, sequence):
        input_tokens   = self.tokenizer(sequence, return_tensors='pt')
        output_tensors = self.embedding_model(**input_tokens)
        return output_tensors
    
    def get_input_seq_ids(self, sequence):
        return self.tokenizer(sequence, return_tensors='pt')
    
    def get_input_tokens_embedding(self, input_tokens):
        return self.embedding_model(**input_tokens)
    

# Positional Encoding class
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    

    
# # input sequence list
# seqs = [
#     '我的名字叫做黑夜路人', 
#     'My name is Black',
#     "我的nickname叫heiyeluren",
# ]

# # call transformers tokenizer get tokens and token-ids
# tokenizer       = Tokenizer(MODEL_BERT_BASE_ZH)
# input_embedding = InputEmbedding(MODEL_BERT_BASE_CHINESE)
# basic_tokenize  = BasicTokenizer(do_lower_case=True)

# for seq in seqs:
#     # seq tokenize
#     tk = basic_tokenize.tokenize(seq)
#     print(seq, ' => ', tk)
#     tokens = tokenizer.tokenize(seq)
#     print(seq, ' => ', tokens)
#     ids = tokenizer.convert_tokens_to_ids(tokens)
#     print(seq, ' => ', ids)

#     # input embedding
#     seq_ebd = input_embedding.get_seq_embedding(seq)
#     print(seq_ebd[0].shape)
#     print(seq_ebd[0])

#     # position encoding
#     position_encode = PositionalEncoding(768, 5000)
#     enc = position_encode.forward(seq_ebd[0])
#     print(position_encode.pe.shape)
#     print(enc.shape)
#     print(enc)
    
    '''
    OUTPUT:

    我的名字叫做黑夜路人  =>  ['我', '的', '名', '字', '叫', '做', '黑', '夜', '路', '人']
    我的名字叫做黑夜路人  =>  ['我', '的', '名', '字', '叫', '做', '黑', '夜', '路', '人']
    我的名字叫做黑夜路人  =>  [2769, 4638, 1399, 2099, 1373, 976, 7946, 1915, 6662, 782]
    torch.Size([1, 12, 768])
    tensor([[[ 0.2516, -0.0583, -0.5325,  ...,  0.7249,  0.6001, -0.0336],
            [ 0.7068, -0.1989,  0.4480,  ..., -0.7411,  0.1905,  0.0682],
            [ 0.8217,  0.0574,  0.1989,  ...,  0.6556,  1.3528,  0.0403],
            ...,
            [-0.7353,  0.2010,  0.3246,  ..., -0.0128, -0.0505, -0.0613],
            [ 0.2031, -0.2266, -0.0970,  ..., -0.1770,  0.1887,  0.2008],
            [ 0.1988, -0.3459,  0.3694,  ...,  0.5481,  0.3442,  0.6523]]],
        grad_fn=<NativeLayerNormBackward0>)
    torch.Size([5000, 1, 768])
    torch.Size([1, 12, 768])
    tensor([[[ 0.2516,  0.9417, -0.5325,  ...,  1.7249,  0.6001,  0.9664],
            [ 0.7068,  0.8011,  0.4480,  ...,  0.2589,  0.1905,  1.0682],
            [ 0.8217,  1.0574,  0.1989,  ...,  1.6556,  1.3528,  1.0403],
            ...,
            [-0.7353,  1.2010,  0.3246,  ...,  0.9872, -0.0505,  0.9387],
            [ 0.2031,  0.7734, -0.0970,  ...,  0.8230,  0.1887,  1.2008],
            [ 0.1988,  0.6541,  0.3694,  ...,  1.5481,  0.3442,  1.6523]]],
        grad_fn=<AddBackward0>)
    My name is Black  =>  ['my', 'name', 'is', 'black']
    My name is Black  =>  ['m', '##y', 'n', '##a', '##m', '##e', 'i', '##s', 'b', '##l', '##a', '##c', '##k']
    My name is Black  =>  [155, 8179, 156, 8139, 8175, 8154, 151, 8118, 144, 8178, 8139, 8177, 8197]
    torch.Size([1, 6, 768])
    tensor([[[-0.9925, -0.0174, -0.4813,  ...,  1.4459,  0.0567,  0.2005],
            [-0.1046, -0.0234, -0.5389,  ...,  0.6051, -0.3514, -0.2064],
            [ 0.6848,  0.5063, -0.5814,  ...,  1.1677,  0.3210,  0.9840],
            [-0.2763,  0.3281, -0.5557,  ...,  1.2178,  0.0132, -0.0884],
            [ 0.1262, -0.0274, -0.9221,  ...,  0.7354,  0.1747, -0.1549],
            [-0.0064,  0.1313, -0.2257,  ...,  0.7574, -0.3390,  0.7666]]],
        grad_fn=<NativeLayerNormBackward0>)
    torch.Size([5000, 1, 768])
    torch.Size([1, 6, 768])
    tensor([[[-0.9925,  0.9826, -0.4813,  ...,  2.4459,  0.0567,  1.2005],
            [-0.1046,  0.9766, -0.5389,  ...,  1.6051, -0.3514,  0.7936],
            [ 0.6848,  1.5063, -0.5814,  ...,  2.1677,  0.3210,  1.9840],
            [-0.2763,  1.3281, -0.5557,  ...,  2.2178,  0.0132,  0.9116],
            [ 0.1262,  0.9726, -0.9221,  ...,  1.7354,  0.1747,  0.8451],
            [-0.0064,  1.1313, -0.2257,  ...,  1.7574, -0.3390,  1.7666]]],
        grad_fn=<AddBackward0>)
    我的nickname叫heiyeluren  =>  ['我', '的', 'nickname', '叫', 'heiyeluren']
    我的nickname叫heiyeluren  =>  ['我', '的', 'n', '##i', '##c', '##k', '##n', '##a', '##m', '##e', '叫', 'h', '##e', '##i', '##y', '##e', '##l', '##u', '##r', '##e', '##n']
    我的nickname叫heiyeluren  =>  [2769, 4638, 156, 8169, 8177, 8197, 8171, 8139, 8175, 8154, 1373, 150, 8154, 8169, 8179, 8154, 8178, 8207, 8180, 8154, 8171]
    torch.Size([1, 12, 768])
    tensor([[[ 0.3072,  0.4564,  0.2195,  ...,  0.4210,  0.9006, -0.7353],
            [ 0.2861,  0.2719,  0.4985,  ..., -0.6424,  0.0051, -0.2119],
            [ 0.3561,  0.1627,  0.8063,  ...,  0.3221,  1.6861, -0.1767],
            ...,
            [-0.1734,  0.9277,  0.0990,  ...,  1.3096,  0.5076, -0.9764],
            [ 0.0515,  0.6708,  0.0723,  ...,  0.2768, -0.1956, -0.8222],
            [ 0.4782,  0.4222,  0.8446,  ...,  0.3568,  0.9280, -0.6421]]],
        grad_fn=<NativeLayerNormBackward0>)
    torch.Size([5000, 1, 768])
    torch.Size([1, 12, 768])
    tensor([[[ 0.3072,  1.4564,  0.2195,  ...,  1.4210,  0.9006,  0.2647],
            [ 0.2861,  1.2719,  0.4985,  ...,  0.3576,  0.0051,  0.7881],
            [ 0.3561,  1.1627,  0.8063,  ...,  1.3221,  1.6861,  0.8233],
            ...,
            [-0.1734,  1.9277,  0.0990,  ...,  2.3096,  0.5076,  0.0236],
            [ 0.0515,  1.6708,  0.0723,  ...,  1.2768, -0.1956,  0.1778],
            [ 0.4782,  1.4222,  0.8446,  ...,  1.3568,  0.9280,  0.3579]]],
        grad_fn=<AddBackward0>)
    '''



# sys.exit()
