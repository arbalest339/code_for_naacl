'''
Author: your name
Date: 2020-10-21 09:11:24
LastEditTime: 2020-10-21 18:10:33
LastEditors: Please set LastEditors
Description: model defination
FilePath: /code_for_naacl/models/main_model.py
'''
import torch
from torch import embedding, log_softmax, sigmoid, softmax, zeros
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.loss import CrossEntropyLoss
from transformers import BertModel, BertForTokenClassification
from models.aggcn import AGGCN
from models.TransD import TransD
from torchcrf import CRF


class TransModel(nn.Module):
    """Model for TransD training

    Args:
        nn ([type]): [description]
    """

    def __init__(self, flags, bertconfig, gcnopt):
        super(TransModel, self).__init__()
        self.dp_num = len(flags.dp_map)
        self.dp_save_path = flags.dp_embedding_path

        # Pre-trained BERT model
        self.bert = BertModel.from_pretrained(flags.pretrained, config=bertconfig)
        # Dropout to avoid overfitting
        self.dropout = nn.Dropout(flags.dropout_rate)

        # aggcn
        self.aggcn = AGGCN(gcnopt, flags)

        # TransD
        self.transd = TransD(bertconfig.vocab_size, self.dp_num,
                             dim_e=gcnopt.emb_dim, dim_r=flags.dp_dim, p_norm=1, norm_flag=True, margin=flags.margin)
        self.dp_emb = self.transd.rel_embeddings.weight.cpu().detach().numpy()

        # full connection layers
        self.bert2gcn = nn.Linear(bertconfig.hidden_size, gcnopt.emb_dim)

    def forward(self, token, pos, ner, arc, matrix, gold, mask, acc_mask):
        # BERT's last hidden layer
        bert_hidden = self.bert(token, encoder_attention_mask=mask, attention_mask=mask)[0]
        bert_hidden = self.dropout(bert_hidden)     # batch_size, max_length, bert_hidden
        bert_emb = self.bert2gcn(bert_hidden)     # bert fc batch_size, max_length, gcn_input_dim

        # aggcn out
        gcn_hidden, pool_mask = self.aggcn(bert_emb, pos, ner, matrix, mask)  # batch_size, max_length, gcn_input_dim
        gcn_flatten = gcn_hidden.view(-1, gcn_hidden.shape[-1])

        # TransD
        # entity(word) embedding  想用索引，需要先转化为2维再还原回去
        emb_h = arc[:, :, 0].view(-1)
        emb_h = gcn_flatten[emb_h].reshape(gcn_hidden.shape)
        emb_t = arc[:, :, -1].view(-1)
        emb_t = gcn_flatten[emb_t].reshape(gcn_hidden.shape)
        # entity(word), relation(dp) index
        token_flatten = token.view(-1)
        batch_h = arc[:, :, 0].view(-1)
        batch_h = token_flatten[batch_h].reshape(token.shape)
        batch_r = arc[:, :, 1]
        batch_t = arc[:, :, -1].view(-1)
        batch_t = token_flatten[batch_t].reshape(token.shape)
        # TransD loss
        transd_loss = self.transd(emb_h, emb_t, batch_h, batch_t, batch_r, mask)

        return transd_loss

    def encoder(self, batch_r):
        dp_emb = self.transd.encoder(batch_r)
        return dp_emb

    def save_embedding(self):
        dps = [i for i in range(self.dp_num)]
        dps = torch.LongTensor(dps).cuda()
        dp_emb = self.encoder(dps).cpu().detach().numpy()
        np.save(self.dp_save_path, dp_emb)


class SeqModel(nn.Module):
    def __init__(self, flags, bertconfig, gcnopt):
        super(SeqModel, self).__init__()
        self.label_num = len(flags.label_map)
        self.pos_num = len(flags.pos_map)
        self.ner_num = len(flags.ner_map)
        self.dp_path = flags.dp_embedding_path
        self.mid_dim = 300
        bertconfig.num_labels = self.label_num
        bertconfig.return_dict = True
        bertconfig.output_hidden_states = True
        self.dp_path = flags.dp_embedding_path

        self.bn1 = nn.BatchNorm1d(flags.max_length)
        self.bn2 = nn.BatchNorm1d(flags.max_length)

        # Pre-trained BERT model
        self.bert = BertForTokenClassification.from_pretrained(flags.pretrained, config=bertconfig)
        # Dropout to avoid overfitting
        self.dropout1 = nn.Dropout(flags.dropout_rate)
        self.dropout2 = nn.Dropout(flags.dropout_rate)

        # transd
        dp_emb = np.load(self.dp_path)
        # self.transd = nn.Embedding.from_pretrained(torch.from_numpy(dp_emb))
        self.transd = nn.Embedding(dp_emb.shape[0], dp_emb.shape[1])

        self.pos_emb = nn.Embedding(self.pos_num, 50)
        self.ner_emb = nn.Embedding(self.ner_num, 50)

        # lstm
        self.lstm = nn.LSTM(self.mid_dim * 2 + flags.dp_dim + 100, flags.lstm_hidden, flags.n_layers, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(flags.dropout_rate)

        # full connection layers
        self.bert2mid = nn.Linear(bertconfig.hidden_size, self.mid_dim)
        self.final2tag = nn.Linear(flags.lstm_hidden * 2, self.label_num)

        # CRF layer
        self.crf_layer = CRF(self.label_num, batch_first=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, token, pos, ner, dp, head, matrix, gold, mask, acc_mask):
        # BERT's last hidden layer
        bert_hidden_t = self.bert(token, labels=gold, attention_mask=mask).hidden_states[-1]
        bert_hidden_h = self.bert(head, labels=gold, attention_mask=mask).hidden_states[-1]
        bert_hidden_t = self.bn1(bert_hidden_t)
        bert_hidden_h = self.bn2(bert_hidden_h)
        bert_hidden_t = self.bert2mid(bert_hidden_t)
        bert_hidden_h = self.bert2mid(bert_hidden_h)
        bert_hidden_t = self.dropout1(bert_hidden_t)     # batch_size, max_length, bert_hidden
        bert_hidden_h = self.dropout2(bert_hidden_h)

        # fc layer
        # TransD
        dp_emb = self.transd(dp)
        pos_emb = self.pos_emb(pos)
        ner_emb = self.ner_emb(ner)

        # feature concat
        final = torch.cat([bert_hidden_t, dp_emb, bert_hidden_h, pos_emb, ner_emb], dim=-1)

        # lstm
        final, _ = self.lstm(final)
        final = self.dropout3(final)
        logits = self.final2tag(final)

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
        bert_hidden_t = self.bert(token, attention_mask=mask).hidden_states[-1]
        bert_hidden_h = self.bert(head, attention_mask=mask).hidden_states[-1]
        bert_hidden_t = self.bn1(bert_hidden_t)
        bert_hidden_h = self.bn2(bert_hidden_h)
        bert_hidden_t = self.bert2mid(bert_hidden_t)
        bert_hidden_h = self.bert2mid(bert_hidden_h)
        bert_hidden_t = self.dropout1(bert_hidden_t)     # batch_size, max_length, bert_hidden
        bert_hidden_h = self.dropout2(bert_hidden_h)

        # fc layer
        # TransD
        dp_emb = self.transd(dp)
        pos_emb = self.pos_emb(pos)
        ner_emb = self.ner_emb(ner)

        # feature concat
        final = torch.cat([bert_hidden_t, dp_emb, bert_hidden_h, pos_emb, ner_emb], dim=-1)

        # lstm
        final, _ = self.lstm(final)
        final = self.dropout3(final)
        logits = self.final2tag(final)

        # crf decode
        tag_seq = self.crf_layer.decode(logits, mask=mask)
        # tag_seq = torch.max(log_softmax(logits[mask.bool()], dim=-1), dim=-1).indices
        # tag_seq = tag_seq.cpu().detach().numpy().tolist()
        return tag_seq
