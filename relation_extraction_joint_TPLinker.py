# -*- coding: utf-8 -*-            
# @Author : Zhihao Zhang
# @Time : 2023/7/24 9:29
import json

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score  # 实体级别评价指标
from typing import List, Optional
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


bertTokenizer = BertTokenizer.from_pretrained(r"D:\pretrained_models\bert-base-chinese")

train_batch_size = 2
epochs = 20
max_len = 512
drop_out = 0.1

bert_lr = 1.0e-5
down_stream_lr = 1.0e-3


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class MyDataset(Dataset):
    def __init__(self, mode):
        self.all_text, self.all_labels = self._read_data(mode)

    def _read_data(self, mode):
        with open("./data/DuIE2.0/duie_{}.json".format(mode), "r", encoding="utf8") as r:
            for line in r:
                line = json.loads(line)
                print(line)
        return all_text, all_labels

    def __len__(self):
        return len(self.all_text)

    def __getitem__(self, item):
        return self.all_text[item], self.all_labels[item]


bertTokenizer = BertTokenizer.from_pretrained(r"D:\pretrained_models\bert-base-chinese")

