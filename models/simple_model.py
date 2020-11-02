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

        # full connection layers
        self.bert2tag = nn.Linear(bertconfig.hidden_size, self.label_num)

        # CRF layer
        self.crf_layer = CRF(self.label_num, batch_first=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, token, pos, ner, arc, matrix, gold, mask, acc_mask):
        # BERT's last hidden layer
        bert_hidden = self.bert(token, labels=gold, attention_mask=mask)
        # bert_hidden = self.dropout(bert_hidden)     # batch_size, max_length, bert_hidden

        # fc layer
        logits = bert_hidden.logits
        # loss = bert_hidden.loss

        # crf loss
        loss = - self.crf_layer(logits, gold, mask=mask)
        # crf_loss = self.loss(logits.view(-1, self.label_num), gold.view(-1))
        pred = torch.Tensor(self.crf_layer.decode(logits)).cuda()
        # pred = torch.max(log_softmax(logits, dim=-1), dim=-1).indices
        zero = torch.zeros(*gold.shape, dtype=gold.dtype).cuda()
        eq = torch.eq(pred, gold.float())
        acc = torch.sum(eq * acc_mask.float()) / torch.sum(acc_mask.float())
        zero_acc = torch.sum(torch.eq(zero, gold.float()) * mask.float()) / torch.sum(mask.float())

        return loss, acc, zero_acc

    def decode(self, token, pos, ner, arc, matrix, mask):
        token = token.unsqueeze(dim=0)
        pos = pos.unsqueeze(dim=0)
        ner = ner.unsqueeze(dim=0)
        arc = arc.unsqueeze(dim=0)
        matrix = matrix.unsqueeze(dim=0)
        mask = mask.unsqueeze(dim=0)

        bert_hidden = self.bert(token, attention_mask=mask)
        logits = bert_hidden.logits

        # crf decode
        tag_seq = self.crf_layer.decode(logits, mask=mask)[0]
        # tag_seq = torch.max(log_softmax(logits[mask.bool()], dim=-1), dim=-1).indices
        # tag_seq = tag_seq.cpu().detach().numpy().tolist()
        return tag_seq
