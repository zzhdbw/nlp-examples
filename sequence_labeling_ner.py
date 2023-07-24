# -*- coding: utf-8 -*-            
# @Author : Zhihao Zhang
# @Time : 2023/7/24 9:29

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score  # 实体级别评价指标

device = 'cuda' if torch.cuda.is_available() else 'cpu'

labels = ['O', 'B-PER.NOM', 'I-GPE.NAM', 'I-ORG.NAM', 'B-LOC.NAM', 'I-GPE.NOM', 'I-LOC.NAM', 'B-LOC.NOM', 'B-PER.NAM',
          'I-LOC.NOM', 'I-PER.NAM', 'B-GPE.NAM', 'B-ORG.NAM', 'B-GPE.NOM', 'I-ORG.NOM', 'B-ORG.NOM', 'I-PER.NOM']
label2id = dict([(label, i) for i, label in enumerate(labels)])
id2label = dict([(v, k) for k, v in label2id.items()])

bertTokenizer = BertTokenizer.from_pretrained(r"D:\pretrained_models\bert-base-chinese")

lr = 2e-5
train_batch_size = 2
epochs = 20
max_len = 200
drop_out = 0.1


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
        with open("./data/Weibo_NER/{}.txt".format(mode), "r", encoding="utf8") as r:
            all_text, all_labels = [], []
            for line in r.read().split("\n\n"):
                text, labels = [], []
                for i in line.split("\n"):
                    ch, label = i.strip().split("\t")
                    text.append(ch)
                    labels.append(label)
                all_text.append(text)
                all_labels.append(labels)
        return all_text, all_labels

    def __len__(self):
        return len(self.all_text)

    def __getitem__(self, item):
        return self.all_text[item], self.all_labels[item]


bertTokenizer = BertTokenizer.from_pretrained(r"D:\pretrained_models\bert-base-chinese")


def collate_fn(batch):
    all_text, all_labels = zip(*batch)

    batch_token_ids = [bertTokenizer.encode(text, max_length=max_len, truncation=True) for text in all_text]
    text_len = [len(token_ids) for token_ids in batch_token_ids]
    batch_max_len = max(text_len)
    batch_max_len = batch_max_len if batch_max_len <= max_len else max_len
    # pad
    batch_token_ids = [token_ids + [bertTokenizer.pad_token_id] * (batch_max_len - len(token_ids)) if len(
        token_ids) < max_len else token_ids[:batch_max_len] for token_ids in batch_token_ids]

    batch_labels = [[label2id["O"]] + [label2id[label] for label in labels] + [label2id["O"]] for labels in
                    all_labels]
    # pad
    batch_labels = [labels + [label2id["O"]] * (batch_max_len - len(labels))
                    if len(labels) < batch_max_len else labels[:batch_max_len]
                    for labels in batch_labels]

    return torch.LongTensor(batch_token_ids).to(device), \
           torch.LongTensor(batch_labels).to(device)


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


# 定义bert上的模型结构
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(r"D:\pretrained_models\bert-base-chinese")
        self.dropout = nn.Dropout(drop_out)
        self.dense = nn.Linear(self.bert.config.hidden_size, len(label2id))

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch_token_ids, labels=None):
        batch_size, batch_max_len = batch_token_ids.shape

        hidden_states, pooling = self.bert(batch_token_ids, return_dict=False)
        batch_mask = batch_token_ids.gt(0).long()

        output = self.dropout(hidden_states)
        output = self.dense(output)

        if (labels is not None):
            return self.loss_fn(output.reshape(-1, len(label2id)), labels.reshape(-1))
        else:
            return torch.argmax(output, dim=-1)


model = Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    train_bar = tqdm(trainDataLoader)
    for batch_token_ids, batch_labels in train_bar:
        train_bar.set_description_str("epoch:{}".format(epoch))

        loss = model(batch_token_ids, batch_labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_bar.set_postfix(loss=loss.item())
    train_bar.close()

    model.eval()
    dev_bar = tqdm(devDataLoader)
    right, totol = 0, 0
    for batch_token_ids, batch_labels in dev_bar:
        predict = model(batch_token_ids)
        labels, predict = labels.tolist(), predict.tolist()

        score = sum([1 if label == predict else 0 for label, predict in zip(labels, predict)])
        right += score
        totol += len(labels)

        acc = right / totol
        dev_bar.set_postfix(acc=acc)
    dev_bar.close()