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

import spacy

# 下载安装模型命令：python -m spacy download xxxxx (xxxxx = zh_core_web_sm or other)
# 可以使用的模型包括：
# 中文：zh_core_web_sm / zh_core_web_md / zh_core_web_lg / zh_core_web_trf 
# 英文：en_core_web_sm / en_core_web_md / en_core_web_ trf / en_core_web_lg

seqs = [
    '我的名字叫做黑夜路人', 
    'My name is Black',
    "我的nickname叫heiyeluren",
    "😊😁😄😉😆🤝👋",
    "はじめまして",
    "잘 부탁 드립니다",
    "До свидания!",
    '英伟达这家公司的英文名称是Nivdia',
]

sp = spacy.load('zh_core_web_md')
doc = sp('西门子将努力参与中国的三峡工程建设。')
for token in doc:
  print(token.text)

# 把seqs中都Tokenize一遍
for seq in seqs:
    doc = sp(seq)
    for token in doc:
       print(token.text)


