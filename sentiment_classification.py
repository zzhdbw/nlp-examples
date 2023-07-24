# -*- coding: utf-8 -*-            
# @Author : Zhihao Zhang
# @Time : 2023/7/23 21:02
# 该代码用以进行情感二分类训练 以acc作为评价指标
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

label2id = {"negative": 0, "positive": 1}
id2label = {0: "negative", 1: "positive"}

# acc: 94.6  epoch:10 max_len:200
# acc: 94.8  epoch:10 max_len:512

lr = 2e-5
train_batch_size = 6
epochs = 20
max_len = 512
drop_out = 0.1


class MyDataset(Dataset):
    def __init__(self, mode):
        self.labels, self.texts = self._read_data(mode)

    def _read_data(self, mode):
        labels, texts = [], []
        with open("./data/ChnSentiCorp/{}.tsv".format(mode), "r", encoding="utf8") as r:
            for line in r.readlines()[1:]:
                if mode == "dev":
                    _, label, text = line.strip().split("\t")
                else:
                    label, text = line.strip().split("\t")
                labels.append(int(label))
                texts.append(text)
                assert len(labels) == len(texts), "len labels isnt equal len texts!"
        return labels, texts

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.labels[item], self.texts[item]


bertTokenizer = BertTokenizer.from_pretrained(r"D:\pretrained_models\bert-base-chinese")


def collate_fn(batch):
    labels, texts = zip(*batch)
    batch_token_ids = [bertTokenizer.encode(text, max_length=max_len, truncation=True) for text in texts]
    text_len = [len(token_ids) for token_ids in batch_token_ids]
    batch_max_len = max(text_len)
    batch_max_len = batch_max_len if batch_max_len <= max_len else max_len

    batch_token_ids = [token_ids + [bertTokenizer.pad_token_id] * (batch_max_len - len(token_ids)) if len(
        token_ids) < max_len else token_ids[:batch_max_len] for token_ids in batch_token_ids]

    return torch.LongTensor(batch_token_ids).to(device), \
           torch.LongTensor(labels).to(device)


# 定义bert上的模型结构
class Model(nn.Module):
    def __init__(self, pool_method='cls') -> None:
        super().__init__()
        self.pool_method = pool_method
        assert self.pool_method in ['cls', 'mean_pooling',
                                    'max_pooling'], "pool_method mast in ['cls', 'mean_pooling', 'max_pooling']!"

        self.bert = BertModel.from_pretrained(r"D:\pretrained_models\bert-base-chinese")
        self.dropout = nn.Dropout(drop_out)
        self.dense = nn.Linear(self.bert.config.hidden_size, len(label2id))

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch_token_ids, labels=None):
        batch_mask = batch_token_ids.gt(0).long()
        hidden_states, pooling = self.bert(batch_token_ids, batch_mask, return_dict=False)

        if self.pool_method == 'cls':
            pooling = pooling
        elif self.pool_method == 'mean_pooling':
            # mean pooling
            hid = torch.sum(hidden_states * batch_mask[:, :, None], dim=1)
            batch_mask = torch.sum(batch_mask, dim=1)[:, None]
            pooling = hid / batch_mask
        elif self.pool_method == 'max_pooling':
            # max pooling
            hid = hidden_states * batch_mask[:, :, None]  # None的意义是增加一个维度
            pooling = torch.max(hid, dim=1).values

        output = self.dropout(pooling)
        output = self.dense(output)

        if (labels is not None):
            return self.loss_fn(output, labels)
        else:
            return torch.argmax(output, dim=-1)


model = Model(pool_method='cls').to(device)

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
