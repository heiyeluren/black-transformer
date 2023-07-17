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

import sys
import os
import json
import math
import regex
import six
import requests
from builtins import str
from itertools import chain

# Black Tokenizer
class Tokenize(object):

    #
    # Inner function
    #

    # tokenizer = BlackTokenize()
    # 初始化类所需要的基本资源
    def __init__(self):
        
        # base data file
        self._TOKEN_ENCODER_FILE = os.path.join(os.path.dirname(__file__), 'encoder.json')
        self._TOKEN_VOCAB_FILE   = os.path.join(os.path.dirname(__file__), "vocab.bpe")
        self._ENCODER_JSON_URL = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json"
        self._VOCAB_BPE_URL   = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"
        
        # base strucutre
        self._REGEX_PATTERN     = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        self._DEFAULT_ENCODING  = "utf-8"
        self._cache             = {}
        self._byte_encoder      = {}
        self._byte_decoder      = {}
        self._bpe_merges        = {}
        self._bpe_ranks         = {}
        self._encoder           = {}
        self._decoder           = {}
        self._regex_compiled    = None

        # fetch encoder.json and vocab.bpe file
        self._fetch_encode_bpe_file()

        # init tokenizer
        self._bpe_merges        = self._init_bpe_merges()
        self._bpe_ranks         = self._init_bpe_rank(self._bpe_merges, range(0, len(self._bpe_merges)))
        self._encoder           = self._init_encoder()
        self._decoder           = self._init_decoder()       #{v: k for k,v in self._encoder.items()}
        self._byte_encoder      = self._init_bytes_encoder()
        self._byte_decoder      = self._init_byte_decoder()  #{v: k for k,v in self._byte_encoder.items()}
        self._regex_compiled    = self._init_regex_compiled()
 

    # 获取GPT-2官方的encoder.json和vocab.bpe文件
    def _fetch_encode_bpe_file(self, encoder_json_url=None, 
                            vocab_bpe_url=None):
        if encoder_json_url is None:
            encoder_json_url = self._ENCODER_JSON_URL
        if vocab_bpe_url is None:
            vocab_bpe_url = self._VOCAB_BPE_URL
        
        if os.path.exists(self._TOKEN_ENCODER_FILE) \
            and os.path.exists(self._TOKEN_VOCAB_FILE) \
            and os.path.getsize(self._TOKEN_ENCODER_FILE) > 0 \
            and os.path.getsize(self._TOKEN_VOCAB_FILE) > 0:
            return True
        
        encoder_json = requests.get(encoder_json_url)
        if encoder_json.status_code != 200 and len(encoder_json.text) < 0:
            raise Exception("Http get "+ encoder_json_url +" failed.")
            return False
        
        vocab_bpe = requests.get(vocab_bpe_url)
        if vocab_bpe.status_code != 200 and len(vocab_bpe.text) < 0:
            raise Exception("Http get "+ vocab_bpe_url +" failed.")
            return False

        with open(self._TOKEN_ENCODER_FILE, "wb") as f:
            f.write(encoder_json.content)
        
        with open(self._TOKEN_VOCAB_FILE, "wb") as f:
            f.write(vocab_bpe.content)

        return True        
    
    # initialize bpe encoder
    # 读取一个存储了 BPE 编码规则的文件，解析出这些规则，并以列表的形式返回
    def _init_bpe_merges(self):
        if len(self._bpe_merges) > 0: return self._bpe_merges
        with open(self._TOKEN_VOCAB_FILE, "r") as f:
            bpe_lines = f.readlines()    
            sliced = bpe_lines[1:len(bpe_lines)-1]
            bpe_merges = [regex.split(r"(\s+)", s) for s in sliced]
            final_merges = []
            for merge in bpe_merges:
                final_merges.append([m for m in merge if len(m.strip()) > 0])
            # self.bpe_merges = final_merges
            # return self.bpe_merges
            return final_merges
        
    # initialize bpe rank
    # 将 BPE 编码规则转换成一个映射表，该映射表可以将由 BPE 编码规则合并后的词汇转换成它们在 BPE 编码序列中的序号
    # 将BFE编码类似 [['hel', 'lo'], ['wor', 'ld']] 进行函数处理以后生成映射表 { 'hel,lo': 0, 'wor,ld': 1 }
    def _init_bpe_rank(self, x, y):
        if len(self._bpe_ranks) > 0: return self._bpe_ranks
        result = {}
        for i in y:
            key = ','.join(x[i])
            if not isinstance(key, str):
                key = key.decode(self._DEFAULT_ENCODING)
            result[key] = y[i]
        return result    

    # load encoder json file 
    # 将已经训练好且存储在文件中的编码器读取到内存中
    def _init_encoder(self):
        if self._encoder != {}: return self._encoder
        with open(self._TOKEN_ENCODER_FILE, 'r') as f:
            encoder = json.load(f)
            return encoder    
    
    # from encoder trans to decoder
    # 生成一个解码器，即将编码器中的每个编码符号与其对应的原始词汇建立一个反向映射表。类似于把 { 'a': 0, 'b': 1, 'c': 2] 转成 {0: 'a', 1: 'b', 2: 'c'}
    def _init_decoder(self):
        if self._decoder != {}: return self._decoder
        return {v: k for k,v in self._encoder.items()}
    
    # initialize special byte encoder
    # 生成一个编码映射表，该映射表可以将一个整数列表编码成按照特定规则转换后的字符串，指定批量特殊进行编码，结构类似 {'33': '!', '34': '"', '35': '#' ... }
    def _init_bytes_encoder(self):
        if self._byte_encoder != {}: return self._byte_encoder
        # include char：
        # !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~
        # ¡¢£¤¥¦§¨©ª«¬ 
        # ­®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ
        bs = list(chain(self._range(self._ord('!'), self._ord('~') + 1), self._range(self._ord('¡'), self._ord('¬') + 1), self._range(self._ord('®'), self._ord('ÿ') + 1)))
        cs = bs[:]
        n = 0
        b = 0
        while b < 2 ** 8:
            if not b in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1
            b += 1

        cs = list(map(lambda x: six.unichr(x), cs))
        result = {}
        for i in range(len(bs)):
            result[str(bs[i])] = cs[i]
        self._byte_encoder = result
        return self._byte_encoder
    
    # from byte encoder trans to byte decoder
    # 将编码映射表生成一个解码映射表，解码映射表可以将一个整数列表解码成原始的字符串，将 {72: 'H'} 变成 {'H': 72}
    def _init_byte_decoder(self):
        if self._byte_decoder != {}: return self._byte_decoder
        return {v: k for k,v in self._init_bytes_encoder().items()}    

    # use utf-8 encode string trans to bytes array
    # 将一个字符串编码成一个整数列表，其中列表中的每个元素都是字符串 token 中每个字符的 ASCII 码
    def _encode_string(self, token):
        return [str(t) for t in list(bytearray(token.encode(self._DEFAULT_ENCODING)))]

    # 返回一个从 x 开始的整数列表，其中起始数字是 x，最大数字为 y-1
    def _range(self, x, y):
        res = [val for val in range(y)][x:]
        return res

    # use utf-8 ord char 
    # 将一个字符串或者字节串转换成对应的 Unicode 码点数值，用于将字符串或者字节串编码成数字形式
    def _ord(self, x):
        if not isinstance(x, str):
            x = x.decode(self._DEFAULT_ENCODING)
        res = ord(x[0])
        return res

    # Pairwise combination of adjacent characters in a word into one element
    # 将一个单词中相邻的字符两两组合成一个元素
    def _get_pairs(self, word):
        pairs = []
        prev_char = word[0]
        for i in range(1, len(word)):
            ch = word[i]
            pairs.append([prev_char, ch])
            prev_char = ch
        return pairs
    
    # initialize regex compiled
    # 生成一个正则表达式对象，主要拆解这些字符
    '''
        's, 't, 're, 've, 'm, 'll, 'd：连缀词，如 you've、you're、I'd 等；
        \p{L}+：一个或多个连续的 Unicode 字母字符；
        \p{N}+：一个或多个连续的 Unicode 数字字符；
        [^\s\p{L}\p{N}]+：一个或多个连续的非空格、非字母、非数字的 Unicode 字符；
        \s+(?!\S)：一个或多个连续的空格字符，且这些空格字符后面没有非空格字符；
        \s+：一个或多个连续的空格字符
    '''
    def _init_regex_compiled(self):
        if self._regex_compiled != None: return self._regex_compiled
        # self._REGEX_PATTERN     = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        regex_compiled = regex.compile(self._REGEX_PATTERN, regex.UNICODE)
        return regex_compiled
    
    #
    # Core API
    #

    
    # use BPE algorithm encode input word or phrase
    # 使用 BPE（Byte Pair Encoding）算法将输入的单词或词组进行编码, 该方法只负责对单个单词或词组进行 BPE 编码，如果要对一组文本数据进行 BPE 编码，需要调用 _bpe_batch 方法
    """
    BPE 是一种压缩算法，用于将文本数据中常见的连续字符序列合并成单个字符，以减少词汇量并提高压缩效率

    1. 基于训练数据生成 BPE 码表，即生成常见字母或字符串的组合，并给组合编码一个整数作为标识符。
    2. 将文本中所有的单词划分成字符或者字符组成的子串。
    3. 在所有单词中找出出现次数最多的字符或者字符组合，将这个字符或者字符组合当做一个新的字符来替代原有单词中的这个字符或者字符组合。并在编码表中添加这个字符或者字符组合的编码。
    3. 重复步骤 3 直到达到预设的 BPE 编码次数或者到达最小词频。
    """
    def _bpe(self, token, bpe_ranks):
        if token in self._cache:
            return self._cache[token]
        word = list(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return token

        while True:
            min_pairs = {}
            for pair in pairs:
                pair_key = ','.join(pair)
                rank = bpe_ranks.get(pair_key, float("nan"))
                min_pairs[10e10 if math.isnan(rank) else rank] = pair_key
            bigram = min_pairs[min(map(int, min_pairs.keys()))]
            if not bigram in bpe_ranks:
                break
            bigram = bigram.split(',', 1)
            first = bigram[0]
            second = bigram[1]
            new_word = []
            i = 0

            while i < len(word):
                j = -1
                try:
                    j = word.index(first, i)
                except:
                    pass
                if j == -1:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        
        word = ' '.join(word)
        self._cache[token] = word
        return word
    
    # use BPE algorithm encode input word or phrase (batch)
    # 批量处理 bpe 编码 (可以传递多个token进行编码)
    def _bpe_batch(self, tokens, bpe_ranks):
        result = []
        for token in tokens:
            if token in self._cache:
                result.append(self._cache[token])
            else:
                code = self._bpe(token, bpe_ranks)
                result.append(code)
                self._cache[token] = code
        return result

    # API: BlackTokenizer.encode(text)
    # 将一个字符串编码成一个整数列表（tokens）
    def encode(self, text):
        """ 
            Transforms a string into an array of tokens 
            :param text: string to be encoded
            :type text: str
            :returns: an array of ints (tokens)
        """
        if not isinstance(text, str):
            text = text.decode(self._DEFAULT_ENCODING)    
        bpe_tokens = []
        matches = self._regex_compiled.findall(text)
        for token in matches:
            token = ''.join([self._byte_encoder[x] for x in self._encode_string(token)])
            new_tokens = [self._encoder[x] for x in self._bpe(token, self._bpe_ranks).split(' ')]
            bpe_tokens.extend(new_tokens)
        return bpe_tokens

    # API: BlackTokenizer.decode(tokens)
    # 将输入的整数列表 tokens 转换成原始字符串
    def decode(self, tokens):
        """ 
            Transforms back an array of tokens into the original string
            :param tokens: an array of ints
            :type tokens: list
            :returns: the original text which was encoded before
        """
        text = ''.join([self._decoder[x] for x in tokens])
        textarr = [int(self._byte_decoder[x]) for x in list(text)]
        text = bytearray(textarr).decode("utf-8")
        return text

    # API: BlackTokenizer.count_tokens(text)
    # 
    def count_tokens(self, text):
        """ 
            Returns an integer representing the tokens count of a given string 
            :param text: string to count tokens from
            :type text: str
            :returns: int representing the tokens count
            
        """
        encoded = self.encode(text)
        return len(encoded)  

    # API: BlackTokenizer.get_tokens(text)
    # 从输入的字符串 text 中获取 tokens 列表。 tokens 列表是由一个或多个单词/字符组成的列表
    def get_token_list(self, text):
        """ 
            Returns an array of tokens from a given string 
            :param text: string to get tokens from
            :type text: str
            :returns: an array of tokens
        """
        if not isinstance(text, str):
            text = text.decode(self._DEFAULT_ENCODING)
        bpe_tokens = []
        matches = self._regex_compiled.findall(text)
        for token in matches:
            new_tokens = [x for x in token.split()]
            bpe_tokens.extend(new_tokens)
        return bpe_tokens            
           
   
'''
Black Tokenizer Test code

Time: 2023/4/29
'''

# seqs = [
#     '我的名字叫做黑夜路人', 
#     'My name is Black',
#     "我的nickname叫heiyeluren",
#     "はじめまして",
#     "잘 부탁 드립니다",
#     "До свидания!",
#     "😊😁😄😉😆🤝👋",
#     "今天的状态很happy，表情是😁",
# ]

# print('\n------------------BlackTokenize Test------------------')

# tk = Tokenize()
# for seq in seqs:
#     token_list = tk.get_token_list(seq)
#     # print('Text:', seq, ' => Tokens:', tokens)
#     enc_seq = tk.encode(seq)
#     # continue
#     dec_seq = tk.decode(enc_seq)
#     token_count = tk.count_tokens(seq)
#     print( 'RawText:', seq, ' => TokenList:', token_list, ' => TokenIDs', enc_seq, ' => TokenCount:', token_count, '=> DecodeText:', dec_seq)

# print('------------------BlackTokenize Test------------------\n')

