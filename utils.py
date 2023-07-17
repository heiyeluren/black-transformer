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

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import os
import copy
import importlib
import spacy


# clones函数的功能是克隆层，不过这些克隆出来的层之间的参数是不共享的
def clones(module, N):
    "生成N个相同的层"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# get_clones函数的功能是克隆层，不过这些克隆出来的层之间的参数是不共享的
def get_clones(module, N):
    "生成N个相同的层"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

