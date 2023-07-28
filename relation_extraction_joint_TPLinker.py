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

labels = [
    json.loads(line)["subject_type"] + "@" + json.loads(line)["predicate"] + "@" + json.loads(line)["object_type"][
        "@value"] for line in open("./data/DuIE2.0/duie_schema.json", "r", encoding="utf8")]
label2id = dict([(label, index) for index, label in enumerate(labels)])
id2label = dict([(index, label) for index, label in enumerate(labels)])


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
        self.texts, self.spo_lists = self._read_data(mode)

    def _read_data(self, mode):
        with open("./data/DuIE2.0/duie_{}.json".format(mode), "r", encoding="utf8") as r:
            texts, spo_lists = [], []
            for line in r:
                line = json.loads(line)
                text = line["text"]
                spo_list = [(spo["subject"],
                             spo["subject_type"] + "@" + spo["predicate"] + "@" + spo["object_type"]['@value'],
                             spo["object"]['@value'])
                            for spo in line["spo_list"]]
                texts.append(text)
                spo_lists.append(spo_list)
        return texts, spo_lists

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return self.texts[item], self.spo_lists[item]


bertTokenizer = BertTokenizer.from_pretrained(r"D:\pretrained_models\bert-base-chinese")


def collate_fn(batch):
    labels, texts = zip(*batch)
    batch_token_ids = [bertTokenizer.encode(text, max_length=max_len, truncation=True) for text in texts]
    text_len = [len(token_ids) for token_ids in batch_token_ids]
    batch_max_len = max(text_len)
    batch_max_len = batch_max_len if batch_max_len <= max_len else max_len

    # pad
    batch_token_ids = [token_ids + [bertTokenizer.pad_token_id] * (batch_max_len - len(token_ids)) if len(
        token_ids) < max_len else token_ids[:batch_max_len] for token_ids in batch_token_ids]

    batch_entity_labels = torch.zeros((len(batch), batch_max_len), dtype=torch.long, device=device)
    batch_head_labels = torch.zeros((len(batch), len(label2id), batch_max_len), dtype=torch.long, device=device)
    batch_tail_labels = torch.zeros((len(batch), len(label2id), batch_max_len), dtype=torch.long, device=device)

    # todo

    return torch.LongTensor(batch_token_ids).to(device), \
           torch.LongTensor(labels).to(device)


TrainDataset = MyDataset("train")
for i in TrainDataset:
    print(i)
