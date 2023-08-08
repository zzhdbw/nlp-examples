# -*- coding: utf-8 -*-
# @Author : Zhihao Zhang
# @Time : 2023/7/28 22:17

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
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch.autograd import Variable

device = 'cuda' if torch.cuda.is_available() else 'cpu'

label2id = {"negative": 0, "positive": 1}
id2label = {0: "negative", 1: "positive"}

lr = 1e-5
train_batch_size = 6
epochs = 20
max_len = 512
drop_out = 0.1
hidden_dim = 300

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
                assert len(labels) == len(texts) and len(texts) == len(heads), \
                    "len labels isnt equal len texts!"
        return labels, texts, heads

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.labels[item], self.texts[item], self.heads[item]


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs.  from AGGCN"""

    def __init__(self, dropout, mem_dim, layers):
        super(GraphConvLayer, self).__init__()

        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = nn.Dropout(dropout)

        # dcgcn layer
        self.Linear = nn.Linear(self.mem_dim, self.mem_dim)

        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        self.weight_list = self.weight_list.cuda()
        self.Linear = self.Linear.cuda()

    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)  # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))
        gcn_ouputs = torch.cat(output_list, dim=2)
        gcn_ouputs = gcn_ouputs + gcn_inputs

        out = self.Linear(gcn_ouputs)

        return out


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """

    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def head_to_tree(head, len_):
    """
    Convert a sequence of head indexes into a tree object.
    """
    head = head[:len_].tolist()
    root = None

    nodes = [Tree() for _ in head]

    for i in range(len(nodes)):
        h = head[i]

        nodes[i].idx = i
        nodes[i].dist = -1  # just a filler
        if h == 0:
            root = nodes[i]
        else:
            nodes[h - 1].add_child(nodes[i])

    assert root is not None
    return root


def tree_to_adj(sent_len, tree, directed=True):
    """
    Convert a tree object to an (numpy) adjacency matrix.将树对象转换为（NumPy）邻接矩阵
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    return ret


# 定义bert上的模型结构
class Model(nn.Module):
    def __init__(self, we, hidden_dim) -> None:
        super().__init__()
        wordEmbedding = we.get_embedding()
        self.embedding = torch.nn.Embedding(wordEmbedding.shape[0], wordEmbedding.shape[1])
        self.embedding.weight.data = wordEmbedding

        self.conv1 = GraphConvLayer(drop_out, wordEmbedding.shape[1], 2)
        self.conv2 = GraphConvLayer(drop_out, wordEmbedding.shape[1], 2)

        self.dropout = nn.Dropout(drop_out)
        self.dense = nn.Linear(wordEmbedding.shape[1], len(label2id))

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch_token_ids, batch_edge_index, batch_edge_lens, batch_labels=None):
        # token_mask
        batch_token_mask = batch_token_ids.gt(0).long()

        maxlen = max(batch_edge_lens)

        def inputs_to_tree_reps(head, batch_edge_lens):
            trees = [head_to_tree(head[i], batch_edge_lens[i]) for i in range(len(batch_edge_lens))]
            adj = [tree_to_adj(maxlen, tree, directed=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj).to(device)

        adj = inputs_to_tree_reps(batch_edge_index.data, batch_edge_lens)

        hidden_states = self.embedding(batch_token_ids)
        hidden_states = self.dropout(hidden_states)

        out = hidden_states
        out = self.conv1(adj, out)
        out = self.conv2(adj, out)

        # mean pooling
        hid = torch.sum(out * batch_token_mask[:, :, None], dim=1)
        batch_mask = torch.sum(batch_token_mask, dim=1)[:, None]
        pooling = hid / batch_mask

        # output = self.dense(pooling)
        output = self.dense(pooling)

        if (batch_labels is not None):
            return self.loss_fn(output, batch_labels)
        else:
            return torch.argmax(output, dim=-1)


model = Model(we, hidden_dim).to(device)

trainDataset = MyDataset("train")
devDataset = MyDataset("dev")


def collate_fn(batch):
    batch_labels, texts, batch_heads = zip(*batch)

    # tokens
    batch_token_ids = [[we.word2index.get(word, we.word2index["PAD"]) for word in text] for text in texts]

    text_len = [len(token_ids) for token_ids in batch_token_ids]
    batch_max_len = max(text_len) if max(text_len) <= max_len else max_len

    batch_token_ids = [token_ids + [we.word2index["PAD"]] * (batch_max_len - len(token_ids)) if len(
        token_ids) < max_len else token_ids[:batch_max_len] for token_ids in batch_token_ids]

    # 边长
    batch_edge_lens = [len(heads) for heads in batch_heads]
    batch_max_edge_len = max(batch_edge_lens) if max(batch_edge_lens) <= max_len else max_len

    # 边 + pad
    batch_edge_index = [
        heads + [0] * (batch_max_edge_len - len(heads))
        if len(heads) < batch_max_edge_len else heads[:batch_max_edge_len]
        for heads in batch_heads
    ]

    return torch.LongTensor(batch_token_ids).to(device), \
           torch.LongTensor(batch_labels).to(device), \
           torch.LongTensor(batch_edge_index).to(device), \
           batch_edge_lens


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
    for batch_token_ids, batch_labels, batch_edge_index, batch_edge_lens in train_bar:
        train_bar.set_description_str("epoch:{}".format(epoch + 1))

        loss = model(batch_token_ids, batch_edge_index, batch_edge_lens, batch_labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_bar.set_postfix(loss=loss.item())
    train_bar.close()

    model.eval()
    dev_bar = tqdm(devDataLoader)
    right, totol = 0, 0
    for batch_token_ids, batch_labels, batch_edge_index, batch_edge_lens in dev_bar:
        predict = model(batch_token_ids, batch_edge_index, batch_edge_lens)

        batch_labels, predict = batch_labels, predict.tolist()

        score = sum([1 if l == p else 0 for l, p in zip(batch_labels, predict)])
        right += score
        totol += len(batch_edge_lens)

        acc = right / totol
        dev_bar.set_postfix(acc=acc)
    dev_bar.close()
