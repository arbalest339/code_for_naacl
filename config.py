'''
Author: your name
Date: 2020-09-23 09:23:31
LastEditTime: 2020-10-21 17:58:33
LastEditors: Please set LastEditors
Description: code and model configs
FilePath: /entity_disambiguation/config.py
'''
import os
import nni
import json
import torch
import argparse


class Flags(object):
    def __init__(self):
        # task info
        self.language = "en"    # en, zh
        self.task = "ore"    # dp_emb, ore, oie
        self.model_type = "bert"   # cnn_lstm, bert
        self.is_continue = False
        self.is_test = False
        self.Canonicalizing = False

        # data dirs
        curpath = os.path.abspath(os.path.dirname(__file__))

        self.pretrained_dir = os.path.join(curpath, f"pretrained/{self.language}_model")    # Path of pretrained ML model
        self.pretrained_config = os.path.join(self.pretrained_dir, "config.json")
        self.pretrained_model = os.path.join(self.pretrained_dir, "pytorch_model.bin")
        self.pretrained_vocab = os.path.join(self.pretrained_dir, "vocab.txt")
        self.embedding_path = self.pretrained_dir = os.path.join(curpath, f"pretrained/wiki.{self.language}.align.vec")

        self.checkpoint_dir = os.path.join(curpath, "checkpoints")  # Path of model checkpoints
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.task}_{self.language}.pkl")
        self.dp_checkpoint_path = os.path.join(self.checkpoint_dir, f"dp_emb_{self.language}.pkl")
        self.dp_embedding_path = os.path.join(self.checkpoint_dir, f"dp_emb_{self.language}(4-128).npy")

        self.data_dir = os.path.join(curpath, f"{self.language}_data")  # Path of input data dir
        # self.data_path = os.path.join(self.data_dir, f"{self.task}_data_{self.language}.json")
        # self.matrix_path = os.path.join(self.data_dir, f"{self.task}_dp_matrix_{self.language}.json")
        self.train_path = os.path.join(self.data_dir, f"{self.task}_train_{self.language}.json")     # !!!!!!!!!!!!!!!!!!!!!!
        self.dev_path = os.path.join(self.data_dir, f"{self.task}_dev_{self.language}.json")
        self.test_path = os.path.join(self.data_dir, f"{self.task}_test_{self.language}.json")
        self.train_mat = os.path.join(self.data_dir, f"{self.task}_train_matrixs_{self.language}.npy")
        self.dev_mat = os.path.join(self.data_dir, f"{self.task}_dev_matrixs_{self.language}.npy")
        self.test_mat = os.path.join(self.data_dir, f"{self.task}_test_matrixs_{self.language}.npy")

        self.out_dir = os.path.join(curpath, "out")  # Path of output results dir

        # train hyper parameters
        self.learning_rate = 3.e-5
        self.epoch = 30
        self.batch_size = 20
        self.test_batch_size = 8
        self.max_length = 128 if self.language == "zh" else 40
        self.dropout_rate = 0.5
        self.weight_decay = 1.e-3
        self.patient = 5
        self.use_cuda = True

        # TransD config
        self.dp_dim = 100
        self.margin = 4.0

        # lstm
        self.lstm_hidden = 300
        self.n_layers = 2

        # global datas
        if self.task == "ore":
            self.label_map = {"O": 0, "B-E1": 1, "I-E1": 2, "B-E2": 3, "I-E2": 4, "B-R": 5, "I-R": 6}
        elif self.task == "oie":
            self.label_map = {"O": 0, "B-E": 1, "I-E": 2, "B-A": 3, "I-A": 4, "B-P": 5, "I-P": 6}
        else:
            self.label_map = {"O": 0, "B-E": 1, "I-E": 2, "B-A": 3, "I-A": 4, "B-P": 5, "I-P": 6, "B-E1": 1, "I-E1": 2, "B-E2": 3, "I-E2": 4, "B-R": 5, "I-R": 6}
        self.dp_map = json.load(open(os.path.join(self.data_dir, "dp_map.json")))
        self.ner_map = json.load(open(os.path.join(self.data_dir, "ner_map.json")))
        self.pos_map = json.load(open(os.path.join(self.data_dir, "pos_map.json")))


FLAGS = Flags()


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=300, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of AGGCN blocks.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.5, help='AGGCN layer dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.set_defaults(lower=False)

parser.add_argument('--heads', type=int, default=3, help='Num of heads in multi-head attention.')
parser.add_argument('--sublayer_first', type=int, default=2, help='Num of the first sublayers in dcgcn block.')
parser.add_argument('--sublayer_second', type=int, default=4, help='Num of the second sublayers in dcgcn block.')

parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type. Default max.')
parser.add_argument('--pooling_l2', type=float, default=0.002, help='L2-penalty for all pooling output.')
parser.add_argument('--mlp_layers', type=int, default=1, help='Number of output mlp layers.')
parser.add_argument('--no_adj', dest='no_adj', action='store_true', help="Zero out adjacency matrix for ablation.")

parser.add_argument('--rnn', dest='rnn', action='store_true', default=False, help='Do not use RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=300, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.5, help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=0.7, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='sgd', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=50, help='Training batch size.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', default=False, help='Ignore CUDA.')

parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

aggcnargs = parser.parse_args()
FLAGS.lr = aggcnargs.lr
