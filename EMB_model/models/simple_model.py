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
        self.bn = nn.BatchNorm1d(flags.max_length)

        # Pre-trained BERT model
        self.bert = BertForTokenClassification(bertconfig)
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(flags.dropout_rate)

        # aggcn
        self.aggcn = AGGCN(gcnopt, flags)

        # full connection layers
        self.bert2tag = nn.Linear(gcnopt.emb_dim, self.label_num)

        # CRF layer
        self.crf_layer = CRF(self.label_num, batch_first=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, token, pos, ner, dp, head, matrix, gold, mask, acc_mask):
        # BERT's last hidden layer
        bert_hidden = self.bert(token, labels=gold, attention_mask=mask).hidden_states[-1]
        # bert_hidden = self.dropout(bert_hidden)     # batch_size, max_length, bert_hidden
        gcn_hidden, pool_mask = self.aggcn(bert_hidden, pos, ner, matrix, mask)

        # fc layer
        # TransD
        # batch_r = arc[:, :, 1]
        # dp_emb = self.transd(batch_r)

        # feature concat
        # logits = torch.cat([bert_hidden, dp_emb], dim=-1)
        logits = self.bert2tag(gcn_hidden)

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
        # bert_hidden = self.dropout(bert_hidden)     # batch_size, max_length, bert_hidden
        gcn_hidden, pool_mask = self.aggcn(bert_hidden, pos, ner, matrix, mask)

        # fc layer
        # TransD
        # batch_r = arc[:, :, 1]
        # dp_emb = self.transd(batch_r)

        # feature concat
        # logits = torch.cat([bert_hidden, dp_emb], dim=-1)
        logits = self.bert2tag(gcn_hidden)

        # crf decode
        tag_seq = self.crf_layer.decode(logits, mask=mask)
        # tag_seq = torch.max(log_softmax(logits[mask.bool()], dim=-1), dim=-1).indices
        # tag_seq = tag_seq.cpu().detach().numpy().tolist()
        return tag_seq
