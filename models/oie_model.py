'''
Author: your name
Date: 2020-11-01 08:57:41
LastEditTime: 2021-04-26 11:47:55
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /code_for_naacl/models/oie_model.py
'''

import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss
from transformers import BertForTokenClassification
from models.aggcn import AGGCN
from models.TransD import TransD
from models.attention import BasicAttention
from torchcrf import CRF


class SeqModel(nn.Module):
    def __init__(self, flags, bertconfig, gcnopt):
        super(SeqModel, self).__init__()
        self.label_num = len(flags.label_map)
        bertconfig.num_labels = self.label_num
        bertconfig.return_dict = True
        bertconfig.output_hidden_states = True
        self.dp_path = flags.dp_embedding_path
        self.features = flags.features
        self.fusion = flags.fusion

        self.act = nn.Sigmoid()
        self.ln = nn.LayerNorm(bertconfig.hidden_size)
        # self.bn = nn.BatchNorm1d(flags.max_length)

        self.bert = BertForTokenClassification.from_pretrained(
            flags.pretrained, config=bertconfig)
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(flags.dropout_rate)

        # aggcn
        # self.bert2label = nn.Linear(bertconfig.hidden_size + flags.dp_dim, self.label_num)
        # self.bert2gcn = nn.Linear(bertconfig.hidden_size, gcnopt.emb_dim)
        # self.aggcn = AGGCN(gcnopt, flags)

        # features
        if "pos" in self.features:
            self.posEmb = nn.Embedding(len(flags.pos_map), flags.feature_dim)
            # self.posAtt = BasicAttention(bertconfig.hidden_size, flags.feature_dim, flags.feature_dim)
        if "ner" in self.features:
            self.nerEmb = nn.Embedding(len(flags.ner_map), flags.feature_dim)
            # self.nerAtt = BasicAttention(bertconfig.hidden_size, flags.feature_dim, flags.feature_dim)
        if "dp" in self.features:
            dpEmb = np.load(self.dp_path)
            if flags.use_transd:
                self.dpEmb = nn.Embedding.from_pretrained(
                    torch.from_numpy(dpEmb))
            else:
                self.dpEmb = nn.Embedding(len(flags.dp_map), flags.feature_dim)
            # self.dpAtt = BasicAttention(bertconfig.hidden_size, flags.feature_dim, flags.feature_dim)
        if self.fusion == "att":
            att_dim = bertconfig.hidden_size + \
                len(self.features)*flags.feature_dim
            self.featureAtt = BasicAttention(att_dim, att_dim, att_dim)
        if self.fusion == "lstm":
            lstm_dim = bertconfig.hidden_size + len(self.features)*flags.feature_dim
            self.bilstm_layer = nn.LSTM(
                lstm_dim, lstm_dim//2, 2, batch_first=True, bidirectional=True)

        # full connection layers
        # self.gcn2tag = nn.Linear(gcnopt.emb_dim + flags.dp_dim, self.label_num)
        self.att2label = nn.Linear(
            bertconfig.hidden_size + len(self.features)*flags.feature_dim, self.label_num)

        # CRF layer
        self.crf_layer = CRF(self.label_num, batch_first=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, token, pos, ner, dp, head, matrix, gold, mask, acc_mask):
        # BERT's last hidden layer
        bert_hidden = self.bert(
            token, labels=gold, attention_mask=mask).hidden_states[-1]
        # bert_hidden = self.ln(bert_hidden)
        # bert_hidden = self.dropout(bert_hidden)     # batch_size, max_length, bert_hidden

        # gcn_input = self.bert2gcn(bert_hidden)
        # gcn_hidden, pool_mask = self.aggcn(gcn_input, pos, ner, matrix, mask)

        feature = None
        # fc layer
        # co-att
        if "pos" in self.features:
            pos_emb = self.posEmb(pos)
            # pos_att = self.posAtt(bert_hidden, pos_emb, pos_emb, mask)
            # logits = torch.cat([logits, pos_emb], dim=-1)
            feature = pos_emb
        if "ner" in self.features:
            ner_emb = self.nerEmb(ner)
            # ner_att = self.nerAtt(bert_hidden, ner_emb, ner_emb, mask)
            # logits = torch.cat([logits, ner_emb], dim=-1)
            if "pos" not in self.features:
                feature = ner_emb
            else:
                feature = torch.cat([feature, ner_emb], dim=-1)
        if "dp" in self.features:
            dp_emb = self.dpEmb(dp)
            # dp_att = self.dpAtt(bert_hidden, dp_emb, dp_emb, mask)
            # logits = torch.cat([logits, dp_emb], dim=-1)
            if "pos" not in self.features and "ner" not in self.features:
                feature = dp_emb
            else:
                feature = torch.cat([feature, dp_emb], dim=-1)

        logits = torch.cat([bert_hidden, feature], dim=-1)
        if self.fusion == "att":
            # logits = torch.cat([bert_hidden, feature], dim=-1)
            logits = self.featureAtt(logits, logits, logits, mask)
        if self.fusion == "lstm":
            # logits = torch.cat([bert_hidden, feature], dim=-1)
            logits, _ = self.bilstm_layer(logits)

            # logits = self.gcn2tag(logits)
        logits = self.dropout(logits)
        logits = self.att2label(logits)

        # crf loss
        loss = - self.crf_layer(logits, gold, mask=mask, reduction="mean")
        # crf_loss = self.loss(logits.view(-1, self.label_num), gold.view(-1))
        pred = torch.Tensor(self.crf_layer.decode(logits)).cuda()
        # pred = torch.max(log_softmax(logits, dim=-1), dim=-1).indices
        zero = torch.zeros(*gold.shape, dtype=gold.dtype).cuda()
        eq = torch.eq(pred, gold.float())
        acc = torch.sum(eq * acc_mask.float()) / torch.sum(acc_mask.float())
        zero_acc = torch.sum(torch.eq(zero, gold.float())
                             * mask.float()) / torch.sum(mask.float())

        return loss, acc, zero_acc

    def decode(self, token, pos, ner, dp, head, matrix, mask):
        bert_hidden = self.bert(token, attention_mask=mask).hidden_states[-1]
        # bert_hidden = self.ln(bert_hidden)
        # bert_hidden = self.dropout(bert_hidden)     # batch_size, max_length, bert_hidden

        # gcn_input = self.bert2gcn(bert_hidden)
        # gcn_hidden, pool_mask = self.aggcn(gcn_input, pos, ner, matrix, mask)

        feature = None
        # fc layer
        # co-att
        if "pos" in self.features:
            pos_emb = self.posEmb(pos)
            # pos_att = self.posAtt(bert_hidden, pos_emb, pos_emb, mask)
            # logits = torch.cat([logits, pos_emb], dim=-1)
            feature = pos_emb
        if "ner" in self.features:
            ner_emb = self.nerEmb(ner)
            # ner_att = self.nerAtt(bert_hidden, ner_emb, ner_emb, mask)
            # logits = torch.cat([logits, ner_emb], dim=-1)
            if "pos" not in self.features:
                feature = ner_emb
            else:
                feature = torch.cat([feature, ner_emb], dim=-1)
        if "dp" in self.features:
            dp_emb = self.dpEmb(dp)
            # dp_att = self.dpAtt(bert_hidden, dp_emb, dp_emb, mask)
            # logits = torch.cat([logits, dp_emb], dim=-1)
            if "pos" not in self.features and "ner" not in self.features:
                feature = dp_emb
            else:
                feature = torch.cat([feature, dp_emb], dim=-1)
        logits = torch.cat([bert_hidden, feature], dim=-1)
        if self.fusion == "att":
            # logits = torch.cat([bert_hidden, feature], dim=-1)
            logits = self.featureAtt(logits, logits, logits, mask)
        if self.fusion == "lstm":
            # logits = torch.cat([bert_hidden, feature], dim=-1)
            logits, _ = self.bilstm_layer(logits)

        # logits = self.gcn2tag(logits)
        logits = self.dropout(logits)
        logits = self.att2label(logits)

        # crf decode
        tag_seq = self.crf_layer.decode(logits, mask=mask)
        # tag_seq = torch.max(log_softmax(logits[mask.bool()], dim=-1), dim=-1).indices
        # tag_seq = tag_seq.cpu().detach().numpy().tolist()
        return tag_seq
