'''
Author: your name
Date: 2020-11-01 08:57:41
LastEditTime: 2021-04-14 17:30:03
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /code_for_naacl/models/oie_model.py
'''
import torch
from torch import embedding, log_softmax, softmax
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss
from transformers import BertForTokenClassification
from models.aggcn import AGGCN
from models.TransD import TransD
from torchcrf import CRF


class SeqModel(nn.Module):
    def __init__(self, flags, bertconfig, gcnopt):
        super(SeqModel, self).__init__()
        self.label_num = len(flags.label_map)
        self.dp_path = flags.dp_embedding_path
        bertconfig.num_labels = self.label_num
        bertconfig.return_dict = True
        bertconfig.output_hidden_states = True
        self.dp_path = flags.dp_embedding_path

        self.act = nn.Sigmoid()
        self.ln = nn.LayerNorm(bertconfig.hidden_size)
        # self.bn = nn.BatchNorm1d(flags.max_length)

        self.bert = BertForTokenClassification.from_pretrained(flags.pretrained, config=bertconfig)
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(flags.dropout_rate)

        # aggcn
        self.bert2label = nn.Linear(bertconfig.hidden_size + flags.dp_dim, self.label_num)
        # self.bert2gcn = nn.Linear(bertconfig.hidden_size, gcnopt.emb_dim)
        self.aggcn = AGGCN(gcnopt, flags)

        # transD
        dp_emb = np.load(self.dp_path)
        self.transd = nn.Embedding.from_pretrained(torch.from_numpy(dp_emb))
        # self.transd = nn.Embedding(dp_emb.shape[0], dp_emb.shape[1])

        # full connection layers
        self.gcn2tag = nn.Linear(gcnopt.emb_dim + flags.dp_dim, self.label_num)

        # CRF layer
        self.crf_layer = CRF(self.label_num, batch_first=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, token, pos, ner, dp, head, matrix, gold, mask, acc_mask):
        # BERT's last hidden layer
        bert_hidden = self.bert(token, labels=gold, attention_mask=mask).hidden_states[-1]
        # bert_hidden = self.ln(bert_hidden)
        bert_hidden = self.dropout(bert_hidden)     # batch_size, max_length, bert_hidden

        # gcn_input = self.bert2gcn(bert_hidden)
        # gcn_hidden, pool_mask = self.aggcn(gcn_input, pos, ner, matrix, mask)

        # fc layer
        # TransD
        dp_emb = self.transd(dp)

        # feature concat
        logits = torch.cat([bert_hidden, dp_emb], dim=-1)
        # logits = self.gcn2tag(logits)
        logits = self.bert2label(logits)

        # crf loss
        loss = - self.crf_layer(logits, gold, mask=mask, reduction="mean")
        # crf_loss = self.loss(logits.view(-1, self.label_num), gold.view(-1))
        pred = torch.Tensor(self.crf_layer.decode(logits)).cuda()
        # pred = torch.max(log_softmax(logits, dim=-1), dim=-1).indices
        zero = torch.zeros(*gold.shape, dtype=gold.dtype).cuda()
        eq = torch.eq(pred, gold.float())
        acc = torch.sum(eq * acc_mask.float()) / torch.sum(acc_mask.float())
        zero_acc = torch.sum(torch.eq(zero, gold.float()) * mask.float()) / torch.sum(mask.float())

        return loss, acc, zero_acc

    def decode(self, token, pos, ner, dp, head, matrix, mask):
        bert_hidden = self.bert(token, attention_mask=mask).hidden_states[-1]
        # bert_hidden = self.ln(bert_hidden)
        # bert_hidden = self.dropout(bert_hidden)     # batch_size, max_length, bert_hidden

        # gcn_input = self.bert2gcn(bert_hidden)
        # gcn_hidden, pool_mask = self.aggcn(gcn_input, pos, ner, matrix, mask)

        # # fc layer
        # # TransD
        dp_emb = self.transd(dp)

        # # feature concat
        logits = torch.cat([bert_hidden, dp_emb], dim=-1)
        # logits = self.gcn2tag(logits)
        logits = self.bert2label(logits)

        # crf decode
        tag_seq = self.crf_layer.decode(logits, mask=mask)
        # tag_seq = torch.max(log_softmax(logits[mask.bool()], dim=-1), dim=-1).indices
        # tag_seq = tag_seq.cpu().detach().numpy().tolist()
        return tag_seq
