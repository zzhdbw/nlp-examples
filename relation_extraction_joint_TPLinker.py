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
max_len = 200
drop_out = 0.1

bert_lr = 1.0e-5
down_stream_lr = 1.0e-3

labels = [
    json.loads(line)["subject_type"] + "@" + json.loads(line)["predicate"] + "@" + json.loads(line)["object_type"][
        "@value"] for line in open("./data/DuIE2.0/duie_schema.json", "r", encoding="utf8")]
label2id = dict([(label, index) for index, label in enumerate(labels)])
id2label = dict([(index, label) for index, label in enumerate(labels)])

bertTokenizer = BertTokenizer.from_pretrained(r"D:\pretrained_models\bert-base-chinese")


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


def collate_fn(batch):
    batch_texts, batch_spo_lists = zip(*batch)

    batch_token_ids = [bertTokenizer.encode(text, max_length=max_len, truncation=True) for text in batch_texts]
    text_len = [len(token_ids) for token_ids in batch_token_ids]
    batch_max_len = max(text_len)
    batch_max_len = batch_max_len if batch_max_len <= max_len else max_len

    # pad
    batch_token_ids = [token_ids + [bertTokenizer.pad_token_id] * (batch_max_len - len(token_ids)) if len(
        token_ids) < max_len else token_ids[:batch_max_len] for token_ids in batch_token_ids]

    # label
    def trans_ij2k(seq_len, i, j):
        return int(0.5 * (2 * seq_len - i + 1) * i + (j - i))

    map_ij2k = {(i, j): trans_ij2k(batch_max_len, i, j) for i in range(batch_max_len) for j in range(batch_max_len) if j >= i}
    map_k2ij = {v: k for k, v in map_ij2k.items()}

    pair_len = batch_max_len * (batch_max_len + 1) // 2
    batch_entity_labels = torch.zeros((len(batch), pair_len), dtype=torch.long, device=device)
    batch_head_labels = torch.zeros((len(batch), len(label2id), pair_len), dtype=torch.long, device=device)
    batch_tail_labels = torch.zeros((len(batch), len(label2id), pair_len), dtype=torch.long, device=device)
    for i, (token_ids, spo_lists) in enumerate(zip(batch_token_ids, batch_spo_lists)):
        for s, p, o in spo_lists:
            s = bertTokenizer.encode(s, add_special_tokens=False)
            o = bertTokenizer.encode(o, add_special_tokens=False)
            p = label2id[p]

            sh = search(s, token_ids)
            oh = search(o, token_ids)

            if (sh == -1 or oh == -1): continue

            st, ot = sh + len(s) - 1, oh + len(o) - 1
            batch_entity_labels[i, map_ij2k[sh, st]] = 1
            batch_entity_labels[i, map_ij2k[oh, ot]] = 1

            # ?
            if sh <= oh:
                batch_head_labels[i, p, map_ij2k[sh, oh]] = 1
            else:
                batch_head_labels[i, p, map_ij2k[oh, sh]] = 2
            if st <= ot:
                batch_tail_labels[i, p, map_ij2k[st, ot]] = 1
            else:
                batch_tail_labels[i, p, map_ij2k[ot, st]] = 2

    batch_token_ids = torch.LongTensor(batch_token_ids).to(device)

    return batch_token_ids, \
           batch_entity_labels.to(device), \
           batch_head_labels.to(device), \
           batch_tail_labels.to(device)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, conditional_size=False, weight=True, bias=True, norm_mode='normal', **kwargs):
        """layernorm 层，这里自行实现，目的是为了兼容 conditianal layernorm，使得可以做条件文本生成、条件分类等任务
           条件layernorm来自于苏剑林的想法，详情：https://spaces.ac.cn/archives/7124

           :param norm_mode: str, `normal`, `rmsnorm`
        """
        super(LayerNorm, self).__init__()

        # 兼容roformer_v2不包含weight
        if weight:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        # 兼容t5不包含bias项, 和t5使用的RMSnorm
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.norm_mode = norm_mode

        self.eps = eps
        self.conditional_size = conditional_size
        if conditional_size:
            # 条件layernorm, 用于条件文本生成,
            # 这里采用全零初始化, 目的是在初始状态不干扰原来的预训练权重
            self.dense1 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense1.weight.data.uniform_(0, 0)
            self.dense2 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense2.weight.data.uniform_(0, 0)

    def forward(self, hidden_states, cond=None):
        if isinstance(hidden_states, (list, tuple)):  # 兼容以前的久逻辑，后期测试后可删除
            cond = hidden_states[1] if self.conditional_size else None
            hidden_states = hidden_states[0]

        if self.norm_mode == 'rmsnorm':
            # t5使用的是RMSnorm
            variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
            o = (hidden_states.float() * torch.rsqrt(variance + self.eps)).type_as(hidden_states)
        else:
            u = hidden_states.mean(-1, keepdim=True)
            s = (hidden_states - u).pow(2).mean(-1, keepdim=True)
            o = (hidden_states - u) / torch.sqrt(s + self.eps)

        if not hasattr(self, 'weight'):
            self.weight = 1
        if not hasattr(self, 'bias'):
            self.bias = 0

        if self.conditional_size and (cond is not None):
            for _ in range(len(hidden_states.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)
            return (self.weight + self.dense1(cond)) * o + (self.bias + self.dense2(cond))
        else:
            return self.weight * o + self.bias


class TplinkerHandshakingKernel(nn.Module):
    '''Tplinker的HandshakingKernel实现'''

    def __init__(self, hidden_size, shaking_type, inner_enc_type=''):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
        elif shaking_type == "cln_plus":
            self.tp_cln = LayerNorm(hidden_size, conditional_size=hidden_size)
            self.inner_context_cln = LayerNorm(hidden_size, conditional_size=hidden_size)

        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = nn.Parameter(torch.rand(hidden_size))
        elif inner_enc_type == "lstm":
            self.inner_context_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True)

        # 自行实现的用torch.gather方式来做，避免循环，目前只实现了cat方式
        # tag_ids = [(i, j) for i in range(maxlen) for j in range(maxlen) if j >= i]
        # gather_idx = torch.tensor(tag_ids, dtype=torch.long).flatten()[None, :, None]
        # self.register_buffer('gather_idx', gather_idx)

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type="lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim=-2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim=-2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * torch.mean(seqence, dim=-2) + (1 - self.lamtha) * torch.max(seqence, dim=-2)[0]
            return pooling

        if "pooling" in inner_enc_type:
            inner_context = torch.stack([pool(seq_hiddens[:, :i + 1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim=1)
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)

        return inner_context

    def forward(self, seq_hiddens):
        '''
        :param seq_hiddens: (batch_size, seq_len, hidden_size)
        :return: shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        '''
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :]  # ind: only look back
            repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)

            if self.shaking_type == "cat":
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cat_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == "cln":
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
            elif self.shaking_type == "cln_plus":
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                shaking_hiddens = self.inner_context_cln([shaking_hiddens, inner_context])

            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim=1)
        return long_shaking_hiddens

        # def handshaking_kernel(self, last_hidden_state):
        #     '''获取(0,0),(0,1),...,(99,99))对应的序列id
        #     '''
        #     btz, _, hdsz = last_hidden_state.shape
        #     gather_idx = self.gather_idx.repeat(btz, 1, hdsz)
        #     concat_hidden_states = torch.gather(last_hidden_state, dim=1, index=gather_idx)  # [btz, pair_len*2, hdsz]
        #     concat_hidden_states = concat_hidden_states.reshape(btz, -1, 2, hdsz)  # concat方式 [btz, pair_len, 2, hdsz]
        #     shaking_hiddens = torch.cat(torch.chunk(concat_hidden_states, chunks=2, dim=-2), dim=-1).squeeze(-2)  # [btz, pair_len, hdsz*2]
        #     return shaking_hiddens


# 定义bert上的模型结构
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(r"D:\pretrained_models\bert-base-chinese")
        self.combine_fc = nn.Linear(768 * 2, 768)
        self.ent_fc = nn.Linear(768, 2)
        self.head_rel_fc = nn.Linear(768, len(label2id) * 3)
        self.tail_rel_fc = nn.Linear(768, len(label2id) * 3)
        self.handshaking_kernel = TplinkerHandshakingKernel(768, shaking_type='cat')

    def forward(self, inputs):
        hidden_states, pooling = self.bert(inputs)  # [btz, seq_len, hdsz]

        shaking_hiddens = self.handshaking_kernel(hidden_states)  # [btz, pair_len, hdsz]
        ent_shaking_outputs = self.ent_fc(shaking_hiddens)  # [btz, pair_len, 2]

        btz, pair_len = shaking_hiddens.shape[:2]
        head_rel_shaking_outputs = self.head_rel_fc(shaking_hiddens).reshape(btz, -1, pair_len, 3)  # [btz, predicate_num, pair_len, 3]
        tail_rel_shaking_outputs = self.tail_rel_fc(shaking_hiddens).reshape(btz, -1, pair_len, 3)

        return ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs


class MyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_preds, y_trues):
        loss_list = []
        for y_pred, y_true in zip(y_preds, y_trues):
            loss = super().forward(y_pred.view(-1, y_pred.size()[-1]), y_true.view(-1))
            loss_list.append(loss)

        z = (2 * len(label2id) + 1)

        # 不同子任务不同权重
        # total_steps = 6000  # 前期实体识别的权重高一些，建议也可以设置为model.total_steps
        # w_ent = max(1 / z + 1 - model.global_step / total_steps, 1 / z)
        # w_rel = min((len(label2id) / z) * model.global_step / total_steps, (len(label2id) / z))

        w_ent = 1
        w_rel = 1
        loss = w_ent * loss_list[0] + w_rel * loss_list[1] + w_rel * loss_list[2]

        # return {'loss': loss, 'entity_loss': loss_list[0], 'head_loss': loss_list[1], 'tail_loss': loss_list[2]}

        return loss


loss_fn = MyLoss()
model = Model().to(device)

trainDataset = MyDataset("train")

trainDataLoader = DataLoader(trainDataset,
                             shuffle=True,
                             batch_size=train_batch_size,
                             collate_fn=collate_fn)

optimizer = optim.Adam(model.parameters(), lr=bert_lr)

for epoch in range(epochs):
    model.train()
    train_bar = tqdm(trainDataLoader)
    for batch_token_ids, batch_entity_labels, batch_head_labels, batch_tail_labels in trainDataLoader:
        train_bar.set_description_str("epoch:{}".format(epoch + 1))
        ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs = model(batch_token_ids)
        loss = loss_fn([ent_shaking_outputs, head_rel_shaking_outputs, tail_rel_shaking_outputs], [batch_entity_labels, batch_head_labels, batch_tail_labels])

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_bar.set_postfix(loss=loss.item())
    train_bar.close()

    # todo
    # evaluate
