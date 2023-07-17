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

import tiktoken
import gpt3_tokenizer

seqs = [
    '我的名字叫做黑夜路人', 
    'My name is Black',
    "我的nickname叫heiyeluren",
    "😊😁😄😉😆🤝👋",
    "はじめまして",
    "잘 부탁 드립니다",
    "До свидания!"
]


# 循环进行加密和解密
i = 0
encoding = [
    "gpt2",
    "r50k_base",
    "p50k_base",
    "p50k_edit",
    "cl100k_base",
]
for curr_enc in encoding:
    i += 1
    print(i, " - Encoding: ", curr_enc)    

    for seq in seqs:
        enc = tiktoken.get_encoding(curr_enc)
        # 字节对编码过程，我的输出是[xxx, yyyy, zzzz]
        encoding_res = enc.encode(seq)
        print(seq, " => ", encoding_res)
        # 字节对解码过程，解码结果：xxxx
        raw_text = enc.decode(encoding_res)
        print(encoding_res , " => ", raw_text)
    
    print("\n")



for seq in seqs:
    # 字节对编码过程，我的输出是[xxx, yyyy, zzzz]
    token_cnt = gpt3_tokenizer.count_tokens(seq)
    enc = gpt3_tokenizer.encode(seq)
    raw = gpt3_tokenizer.decode(enc)

    print(seq, " => TokenCount", token_cnt, " => ", enc, " => ", raw)


