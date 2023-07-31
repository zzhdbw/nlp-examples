# -*- coding: utf-8 -*-
# @Author : Zhihao Zhang
# @Time : 2023/7/29 10:30

# 先运行 sentiment_classification_w2v_prepare.py

import json
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from stanfordcorenlp import StanfordCoreNLP
from collections import Counter
import numpy as np
import fasttext

device = 'cuda' if torch.cuda.is_available() else 'cpu'

label2id = {"negative": 0, "positive": 1}
id2label = {0: "negative", 1: "positive"}

lr = 2e-4
train_batch_size = 6
epochs = 20
max_len = 200
drop_out = 0.1


class WordEmbedding():
    def __init__(self, train_dataset_path, dev_dataset_path, word_dim):
        self.word_dim = word_dim

        self.words_list = []  # 全部词，有重复

        self._read_text(train_dataset_path)  # 读取训练集
        self._read_text(dev_dataset_path)  # 读取测试集

        self.words, _ = zip(*Counter(self.words_list).most_common())
        self.words = ["PAD", "UNK"] + list(self.words)  # 全部词，无重复

        self.word2index = dict([(word, index) for index, word in enumerate(self.words)])
        self.index2word = dict([(index, word) for index, word in enumerate(self.words)])

    def _read_text(self, dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as r:
            for line in r:
                line = json.loads(line)
                words = line["words"]
                self.words_list.extend(words)

    def get_embedding(self):
        print("load embedding from fasttext")
        fasttext_model = fasttext.load_model("D:\pretrained_models/fasttext_embedding/cc.zh.300.bin")
        print("load embedding down")

        embedding_matrix = np.zeros((len(self.words), self.word_dim))

        embedding_matrix[1] = np.random.random((1, self.word_dim))  # UNK_embedding
        for word in tqdm(self.words[2:]):
            embedding_matrix[self.word2index[word]] = fasttext_model.get_word_vector(word)

        return torch.FloatTensor(embedding_matrix)


we = WordEmbedding("./data/ChnSentiCorp/train.json", "./data/ChnSentiCorp/dev.json", 300)


class MyDataset(Dataset):
    def __init__(self, mode):
        self.labels, self.texts, self.heads = self._read_data(mode)

    def _read_data(self, mode):
        labels, texts, heads = [], [], []
        with open("./data/ChnSentiCorp/{}.json".format(mode), "r", encoding="utf8") as r:
            for line in r:
                line = json.loads(line)

                word = line["words"]
                head = line["heads"]
                label = line["label"]

                labels.append(int(label))
                texts.append(word)
                heads.append(head)
                assert len(labels) == len(texts) and len(texts) == len(heads), "len labels isnt equal len texts!"
        return labels, texts, heads

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.labels[item], self.texts[item], self.heads[item]


def collate_fn(batch):
    labels, texts, heads = zip(*batch)
    batch_token_ids = [[we.word2index.get(word, we.word2index["PAD"]) for word in text] for text in texts]

    text_len = [len(token_ids) for token_ids in batch_token_ids]
    batch_max_len = max(text_len)
    batch_max_len = batch_max_len if batch_max_len <= max_len else max_len

    batch_token_ids = [token_ids + [we.word2index["PAD"]] * (batch_max_len - len(token_ids)) if len(
        token_ids) < max_len else token_ids[:batch_max_len] for token_ids in batch_token_ids]

    return torch.LongTensor(batch_token_ids).to(device), \
           torch.LongTensor(labels).to(device)


# 定义bert上的模型结构
class Model(nn.Module):
    def __init__(self, we) -> None:
        super().__init__()
        wordEmbedding = we.get_embedding()
        self.embedding = torch.nn.Embedding(wordEmbedding.shape[0], wordEmbedding.shape[1])
        self.embedding.weight.data = wordEmbedding

        self.dropout = nn.Dropout(drop_out)
        self.dense = nn.Linear(wordEmbedding.shape[1], len(label2id))

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch_token_ids, labels=None):
        batch_mask = batch_token_ids.gt(0).long()
        hidden_states = self.embedding(batch_token_ids)

        # mean pooling
        hid = torch.sum(hidden_states * batch_mask[:, :, None], dim=1)
        batch_mask = torch.sum(batch_mask, dim=1)[:, None]
        pooling = hid / batch_mask

        output = self.dropout(pooling)
        output = self.dense(output)

        if (labels is not None):
            return self.loss_fn(output, labels)
        else:
            return torch.argmax(output, dim=-1)


model = Model(we).to(device)

trainDataset = MyDataset("train")
devDataset = MyDataset("dev")

trainDataLoader = DataLoader(trainDataset,
                             shuffle=True,
                             batch_size=train_batch_size,
                             collate_fn=collate_fn)
devDataLoader = DataLoader(devDataset,
                           # shuffle=True,
                           batch_size=train_batch_size,
                           collate_fn=collate_fn)

optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):

    model.train()
    train_bar = tqdm(trainDataLoader)
    for batch_token_ids, labels in train_bar:
        train_bar.set_description_str("epoch:{}".format(epoch + 1))

        loss = model(batch_token_ids, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_bar.set_postfix(loss=loss.item())
    train_bar.close()

    model.eval()
    dev_bar = tqdm(devDataLoader)
    right, totol = 0, 0
    for batch_token_ids, labels in dev_bar:
        predict = model(batch_token_ids)
        labels, predict = labels.tolist(), predict.tolist()

        score = sum([1 if label == predict else 0 for label, predict in zip(labels, predict)])
        right += score
        totol += len(labels)

        acc = right / totol
        dev_bar.set_postfix(acc=acc)
    dev_bar.close()
