import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import Counter
import re
import json
import nltk
import jieba
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# 数据清洗函数
def clean_sentence(sentence):
    # 去除非法字符
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # 过滤过长句子（设定长度阈值为100）
    if len(sentence) > 100:
        sentence = sentence[:100]
    return sentence

# 分词
def tokenize_en(sentence):
    return nltk.word_tokenize(sentence)

def tokenize_zh(sentence):
    return jieba.lcut(sentence)

def build_vocab(tokenized_sentences, min_freq=4):
    counter = Counter(token for sentence in tokenized_sentences for token in sentence)
    # 过滤词频小的单词
    filtered_tokens = [token for token, freq in counter.items() if freq >= min_freq]
    vocab = {token: idx for idx, token in enumerate(filtered_tokens)}
    idx2token = {idx: token for idx, token in enumerate(filtered_tokens)}
    vocab['sos'] = len(vocab)
    vocab['eos'] = len(vocab)
    idx2token[len(idx2token)] = 'sos'
    idx2token[len(idx2token)] = 'eos'
    return vocab, idx2token

class enzhDataset(Dataset):
    def __init__(self):
        super(enzhDataset, self).__init__()
        self.train_file_path = './data/train_10k.jsonl'
        self.test_file_path = './data/test.jsonl'

        # nltk.download('punkt')

        self.train_en_sentences = []
        self.train_zh_sentences = []
        with open(self.train_file_path, 'r', encoding='utf-8') as file:
            print("Loading Training Parallel Corpus...")
            for line in tqdm(file, desc="Processing lines"):
                data = json.loads(line)
                en_sentence = clean_sentence(data['en'])
                zh_sentence = clean_sentence(data['zh'])

                en_tokens = tokenize_en(en_sentence)
                zh_tokens = tokenize_zh(zh_sentence)

                self.train_en_sentences.append(en_tokens)
                self.train_zh_sentences.append(zh_tokens)

        self.test_en_sentences = []
        self.test_zh_sentences = []
        with open(self.test_file_path, 'r', encoding='utf-8') as file:
            print("Loading Testing Parallel Corpus...")
            for line in tqdm(file, desc="Processing lines"):
                data = json.loads(line)
                en_sentence = clean_sentence(data['en'])
                zh_sentence = clean_sentence(data['zh'])

                en_tokens = tokenize_en(en_sentence)
                zh_tokens = tokenize_zh(zh_sentence)

                self.test_en_sentences.append(en_tokens)
                self.test_zh_sentences.append(zh_tokens)

        self.en_sentences = self.train_en_sentences + self.test_en_sentences
        self.zh_sentences = self.train_zh_sentences + self.test_zh_sentences
        self.en_vocab, self.en_idx2token = build_vocab(self.en_sentences)
        self.zh_vocab, self.zh_idx2token = build_vocab(self.zh_sentences)

    def __getitem__(self, index, split):
        if split == 'train':
            en_sen = self.train_en_sentences[index]
            en_res = [self.en_vocab[token] for token in en_sen if token in self.en_vocab]
            zh_sen = self.train_zh_sentences[index]
            zh_res = [self.zh_vocab[token] for token in zh_sen if token in self.zh_vocab]
            return torch.tensor(en_res, dtype=torch.long), torch.tensor(zh_res, dtype=torch.long)
        elif split == 'test':
            en_sen = self.test_en_sentences[index]
            en_res = [self.en_vocab[token] for token in en_sen if token in self.en_vocab]
            zh_sen = self.test_zh_sentences[index]
            zh_res = [self.zh_vocab[token] for token in zh_sen if token in self.zh_vocab]
            return torch.tensor(en_res, dtype=torch.long), torch.tensor(zh_res, dtype=torch.long)
        else:
            raise Exception
        
    def __len__(self):
        return len(self.en_sentences)
    
class enzhWrapper(Dataset):
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split
    
    def __getitem__(self, index):
        return self.dataset.__getitem__(index, self.split)
    
    def __len__(self):
        if self.split == 'train':
            return len(self.dataset.train_en_sentences)
        elif self.split == 'test':
            return len(self.dataset.test_en_sentences)