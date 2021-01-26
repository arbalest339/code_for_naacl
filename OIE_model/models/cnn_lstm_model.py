import torch
from torch import embedding, sigmoid, softmax
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from models.aggcn import AGGCN
from torchcrf import CRF


class CnnLSTM(nn.Module):
    def __init__(self, flags, bertconfig, gcnopt):
        super(CnnLSTM, self).__init__()
        self.label_num = len(flags.label_map)
        self.dp_path = flags.dp_embedding_path
        self.token2idx, self.weights = self.load_token_embedding_layer_weight(flags.embedding_path)

        self.act = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(flags.max_length)

        # embedding layers
        self.token_embedding_layer = nn.Embedding.from_pretrained(embeddings=self.weights, freeze=False, padding_idx=0)

        # lstm
        self.lstm_input_size = self.word_embedding_dim
        self.n_lstm_layers = 1
        self.batch_size = flags.batch_size
        self.lstm_hidden_size = 200
        self.max_length = flags.max_length
        self.bilstm_layer = nn.LSTM(self.lstm_input_size,
                                    self.lstm_hidden_size,
                                    self.n_lstm_layers,
                                    batch_first=True,
                                    bidirectional=True)

        # aggcn
        self.aggcn = AGGCN(gcnopt, flags)

        # TransD
        dp_emb = np.load(self.dp_path)
        self.transd = nn.Embedding.from_pretrained(torch.from_numpy(dp_emb))

        # full connection layers
        self.input2lstm = nn.Linear(self.word_embedding_dim, self.lstm_hidden_size)
        self.lstm2gcn = nn.Linear(self.lstm_hidden_size * 2, gcnopt.emb_dim)
        self.final2tag = nn.Linear(gcnopt.emb_dim, self.label_num)

        # CRF layer
        self.crf_layer = CRF(self.label_num, batch_first=True)
        self.loss = nn.CrossEntropyLoss()

        # drop outs
        self.dropout_input = nn.Dropout(flags.dropout_rate)
        self.dropout_lstm = nn.Dropout(flags.dropout_rate)
        self.dropout_gcn = nn.Dropout(flags.dropout_rate)
        self.dropout_transd = nn.Dropout(flags.dropout_rate)

    def forward(self, token, pos, ner, arc, matrix, gold, mask, acc_mask):
        # BERT's last hidden layer
        token_emb = self.token_embedding_layer(token)
        lstm_input = self.dropout_input(token_emb)     # batch_size, max_length, bert_hidden
        # lstm_input = self.input2lstm(lstm_input)     # bert fc batch_size, max_length, gcn_input_dim

        # lstm
        lstm_ouput, _ = self.bilstm_layer(lstm_input)
        lstm_ouput = self.dropout_lstm(lstm_ouput)
        gcn_input = self.lstm2gcn(lstm_ouput)

        # fc layer
        logits = self.final2tag(gcn_input)
        logits = sigmoid(logits)

        # crf loss
        crf_loss = - self.crf_layer(logits, gold, mask=mask)
        pred = torch.Tensor(self.crf_layer.decode(logits)).cuda()
        zero = torch.zeros(*pred.shape, dtype=pred.dtype).cuda()
        eq = torch.eq(pred, gold.float())
        acc_mask = torch.ne(gold.float(), 0)
        acc = torch.sum(eq * acc_mask.float()) / torch.sum(acc_mask.float())
        zero_acc = torch.sum(torch.eq(zero, gold.float()) * mask.float()) / torch.sum(mask.float())
        # crf_loss = self.loss(logits.view(-1, self.label_num), gold.view(-1))

        return crf_loss, acc, zero_acc

    def decode(self, token, pos, ner, arc, matrix, mask):
        token = token.unsqueeze(dim=0)
        pos = pos.unsqueeze(dim=0)
        ner = ner.unsqueeze(dim=0)
        arc = arc.unsqueeze(dim=0)
        matrix = matrix.unsqueeze(dim=0)
        mask = mask.unsqueeze(dim=0)

        token_emb = self.token_embedding_layer(token)
        token_emb = self.dropout(token_emb)     # batch_size, max_length, bert_hidden
        gcn_input = self.input2gcn(token_emb)     # bert fc batch_size, max_length, gcn_input_dim

        # aggcn out
        gcn_hidden, pool_mask = self.aggcn(gcn_input, pos, ner, matrix, mask)  # batch_size, max_length, gcn_input_dim

        # TransD
        batch_r = arc[:, :, 1]
        dp_emb = self.transd(batch_r)

        # feature concat
        final_emb = torch.cat([gcn_hidden, dp_emb], dim=-1)

        # fc layer
        logits = self.final2tag(final_emb)
        logits = self.bn(logits)
        logits = self.act(logits)

        # crf decode
        tag_seq = self.crf_layer.decode(logits, mask=mask)
        return tag_seq

    def load_token_embedding_layer_weight(self, embedding_path):
        f = open(embedding_path)
        token2idx = {}
        weights = []
        self.dict_length, self.word_embedding_dim = next(f).strip().split(" ")
        self.dict_length = int(self.dict_length)
        self.word_embedding_dim = int(self.word_embedding_dim)

        weights.append(np.random.uniform(-0.25, 0.25, self.word_embedding_dim))
        token2idx["[UKN]"] = 0
        weights.append(np.random.uniform(-0.25, 0.25, self.word_embedding_dim))
        token2idx["[PAD]"] = 1

        for i, line in enumerate(f):
            if i > 50000:
                break
            values = line.split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            weights.append(coefs)
            token2idx[word] = i + 2
        f.close()

        print('Number of words with embedding: ', i+2)
        return token2idx, torch.Tensor(weights)
