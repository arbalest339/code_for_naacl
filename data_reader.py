"""
read training/test/predict data
"""
from torch import randint
from config import FLAGS
import json
import torch
import random
import torch.utils.data as data
import numpy as np


class OIEDataset(data.Dataset):
    def __init__(self, data_path, matrix_path, tokenizer, max_length, mode, use_cuda=True):
        self.use_cuda = use_cuda
        self.max_length = max_length - 2
        self.label_map = FLAGS.label_map
        if mode == "train":
            self.load_train(data_path, matrix_path, tokenizer)
        else:
            self.load_test(data_path, matrix_path, tokenizer)

    def load_train(self, data_path, matrix_path, tokenizer):
        self.data = list()
        datas = json.load(open(data_path))
        matrixs = np.load(matrix_path, allow_pickle=True)

        for line, matrix in zip(datas, matrixs):
            # 文件中读取基础数据
            token, pos, ner, gold = \
                line["token"], line["pos"], line["ner"], line["gold"]

            arc = []     # for TransD
            for i, row in enumerate(matrix):
                for j, t in enumerate(row):
                    if int(t) > 0:
                        arc.append([i, int(t), j])
                        matrix[i][j] = 1.0
                    else:
                        matrix[i][j] = 0.0
            arc.sort(key=lambda x: x[2])

            # padding
            token_length = len(token)
            pad_length = self.max_length - token_length
            if pad_length > 0:
                # BERT special token
                token = ["[CLS]"] + token + ["[SEP]"]
                pos = [0] + pos + [0]
                ner = [0] + ner + [0]
                gold = ["O"] + gold + ["O"]
                # padding
                token += ["[PAD]"] * pad_length
                pos += [0] * pad_length
                ner += [0] * pad_length
                gold += ["O"] * pad_length
                # dp arcs pad
                for p in range(pad_length + 2):     # 为arc pad的是随机产生的负样本
                    neg = arc[-1]
                    while neg in arc:
                        neg = [int(random.uniform(1, token_length+1)), int(random.uniform(0, len(FLAGS.dp_map))), int(random.uniform(1, token_length+1))]
                    arc += [neg]
                # mask pad
                mask = [1] * (token_length + 2) + [0] * pad_length
                # matrix pad
                for i, row in enumerate(matrix):
                    matrix[i] = row + [0.0] * (pad_length + 1)
                for i in range(pad_length + 1):
                    matrix.append([0.0] * (self.max_length + 2))
            else:
                token = ["[CLS]"] + token[:self.max_length] + ["[SEP]"]
                pos = [0] + pos[:self.max_length] + [0]
                ner = [0] + ner[:self.max_length] + [0]
                gold = [0] + gold[:self.max_length] + [0]
                arc = arc[:(self.max_length + 2)]
                mask = [1] * (self.max_length + 2)
                matrix = matrix[:self.max_length+2, :self.max_length+2]

            acc_mask = [1 if g != "O" else 0 for g in gold]
            acc_mask[0] = 1

            # 数字化
            token = tokenizer.convert_tokens_to_ids(token)
            gold = [self.label_map[g] for g in gold]

            # tensor化
            token = torch.LongTensor(token).cuda() if self.use_cuda else torch.LongTensor(token)
            pos = torch.LongTensor(pos).cuda() if self.use_cuda else torch.LongTensor(pos)
            ner = torch.LongTensor(ner).cuda() if self.use_cuda else torch.LongTensor(ner)
            gold = torch.LongTensor(gold).cuda() if self.use_cuda else torch.LongTensor(gold)
            arc = torch.LongTensor(arc).cuda() if self.use_cuda else torch.LongTensor(arc)
            mask = torch.BoolTensor(mask).cuda() if self.use_cuda else torch.BoolTensor(mask)
            acc_mask = torch.BoolTensor(acc_mask).cuda() if self.use_cuda else torch.BoolTensor(acc_mask)
            matrix = torch.Tensor(matrix).cuda() if self.use_cuda else torch.Tensor(matrix)

            self.data.append([token, pos, ner, arc, matrix, gold, mask, acc_mask])

    def load_test(self, data_path, matrix_path, tokenizer):
        self.data = list()
        datas = json.load(open(data_path))
        matrixs = np.load(matrix_path, allow_pickle=True)

        for line, matrix in zip(datas, matrixs):
            # 文件中读取基础数据
            if FLAGS.task == "ore":
                token, pos, ner, e1, e2, r = \
                    line["token"], line["pos"], line["ner"], line["e1"], line["e2"], line["pred"]
            else:
                token, pos, ner, e1, e2, r = \
                    line["token"], line["pos"], line["ner"], line["sub"], line["obj"], line["pred"]

            arc = []     # for TransD
            for i, row in enumerate(matrix):
                for j, t in enumerate(row):
                    if int(t) > 0:
                        arc.append([i, int(t), j])
                        matrix[i][j] = 1.0
                    else:
                        matrix[i][j] = 0.0

            # padding
            token_length = len(token)
            pad_length = self.max_length - token_length
            if pad_length > 0:
                # BERT special token
                token = ["[CLS]"] + token + ["[SEP]"]
                pos = [0] + pos + [0]
                ner = [0] + ner + [0]
                # padding
                token += ["[PAD]"] * pad_length
                pos += [0] * pad_length
                ner += [0] * pad_length
                # dp arcs pad
                for p in range(pad_length + 2):     # 为arc pad的是随机产生的负样本
                    neg = arc[-1]
                    while neg in arc:
                        neg = [int(random.uniform(1, token_length+1)), int(random.uniform(0, len(FLAGS.dp_map))), int(random.uniform(1, token_length+1))]
                    arc += [neg]
                # mask pad
                mask = [1] * (token_length + 2) + [0] * pad_length
                # matrix pad
                for i, row in enumerate(matrix):
                    matrix[i] = row + [0.0] * (pad_length + 1)
                for i in range(pad_length + 1):
                    matrix.append([0.0] * (self.max_length + 2))
            else:
                token = ["[CLS]"] + token[:self.max_length] + ["[SEP]"]
                pos = [0] + pos[:self.max_length] + [0]
                ner = [0] + ner[:self.max_length] + [0]
                arc = arc[:(self.max_length + 2)]
                mask = [1] * (self.max_length + 2)
                matrix = matrix[:self.max_length+2, :self.max_length+2]

            # 数字化
            token = tokenizer.convert_tokens_to_ids(token)

            # tensor化
            token = torch.LongTensor(token).cuda() if self.use_cuda else torch.LongTensor(token)
            pos = torch.LongTensor(pos).cuda() if self.use_cuda else torch.LongTensor(pos)
            ner = torch.LongTensor(ner).cuda() if self.use_cuda else torch.LongTensor(ner)
            arc = torch.LongTensor(arc).cuda() if self.use_cuda else torch.LongTensor(arc)
            mask = torch.ByteTensor(mask).cuda() if self.use_cuda else torch.ByteTensor(mask)
            matrix = torch.Tensor(matrix).cuda() if self.use_cuda else torch.Tensor(matrix)

            self.data.append([token, pos, ner, arc, matrix, e1, e2, r, mask])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
