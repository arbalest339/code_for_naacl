import os
import time
import json
import torch
import numpy as np

from transformers import BertTokenizer, BertConfig
# from models.main_model import SeqModel
from models.oie_model import SeqModel
from data_reader import AnalyseDataset
from config import FLAGS, aggcnargs


def en_metrics(e1, e2, r, tag_seq):
    positive_true = 0
    positive_false = 0
    negative_false = 0

    for e1p in e1:
        if tag_seq[e1p] in (1, 2):
            positive_true += 1
        else:
            negative_false += 1

    for e2p in e2:
        if tag_seq[e2p] in (3, 4):
            positive_true += 1
        else:
            negative_false += 1

    for rp in r:
        if tag_seq[rp] in (5, 6):
            positive_true += 1
        else:
            negative_false += 1

    for i, t in enumerate(tag_seq):
        if i not in e1 and t in (1, 2):
            positive_false += 1
        if i not in e2 and t in (3, 4):
            positive_false += 1
        if i not in r and t in (5, 6):
            positive_false += 1

    return positive_true, positive_false, negative_false


def zh_metrics(e1, e2, r, tag_seq):
    positive_true = 0
    positive_false = 0
    negative_false = 0

    for e1p in e1:
        if tag_seq[e1p + 1] in (1, 2):
            positive_true += 1
        else:
            negative_false += 1

    for e2p in e2:
        if tag_seq[e2p + 1] in (3, 4):
            positive_true += 1
        else:
            negative_false += 1

    for rp in r:
        if tag_seq[rp + 1] in (5, 6):
            positive_true += 1
        else:
            negative_false += 1

    for i, t in enumerate(tag_seq):
        if i - 1 not in e1 and t in (1, 2):
            positive_false += 1
        if i - 1 not in e2 and t in (3, 4):
            positive_false += 1
        if i - 1 not in r and t in (5, 6):
            positive_false += 1

    return positive_true, positive_false, negative_false


def test():
    # load from pretrained config file
    # bertconfig = json.load(open(FLAGS.pretrained_config))
    if FLAGS.task == "ore":
        label_map = {0: "O", 1: "B-E1", 2: "I-E1", 3: "B-E2", 4: "I-E2", 5: "B-R", 6: "I-R"}
    else:
        label_map = {0: "O", 1: "B-E", 2: "I-E", 3: "B-A", 4: "I-A", 5: "B-P", 6: "I-P"}
    bertconfig = BertConfig.from_pretrained(FLAGS.pretrained)

    # Initiate model
    print("Initiating model.")
    model = SeqModel(FLAGS, bertconfig, aggcnargs)
    if torch.cuda.is_available():
        model.cuda()

    model.load_state_dict(torch.load(FLAGS.checkpoint_path))
    print('Loading from previous model.')
    print("Model initialized.")

    # load data
    print("Loading traning and valid data")
    tokenizer = BertTokenizer.from_pretrained(FLAGS.pretrained)
    test_set = AnalyseDataset(FLAGS.test_path, FLAGS.test_mat, tokenizer, FLAGS.max_length, task=FLAGS.task)
    wf = open("out/analyse", "w")
    print("Start testing", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    model.eval()
    for token, pos, ner, dp, head, matrix, gold, mask, token_str, e1, e2, r in test_set.data:
        model.zero_grad()
        token = token.unsqueeze(dim=0)
        pos = pos.unsqueeze(dim=0)
        ner = ner.unsqueeze(dim=0)
        dp = dp.unsqueeze(dim=0)
        head = head.unsqueeze(dim=0)
        matrix = matrix.unsqueeze(dim=0)
        mask = mask.unsqueeze(dim=0)
        tag_seq = model.decode(token, pos, ner, dp, head, matrix, mask)[0]
        tag_seq = [label_map[t] for t in tag_seq[1:-1]]
        error_token = []
        wf.write(" ".join(token_str)+"\n")
        wf.write(" ".join(tag_seq)+"\n")
        wf.write(" ".join(gold)+"\n")
        for p, g, t in zip(tag_seq, gold, token_str):
            if p != g:
                error_token.append(str([p, g, t]))
        wf.writelines(error_token)
        wf.write("\n")

    wf.close()


if __name__ == "__main__":
    test()
