import json
from os import pread
import torch
import numpy as np
from config import FLAGS, aggcnargs
from transformers import BertConfig
from models.main_model import TransModel

oie_train_path = "zh_data/oie_train_zh.json"
ore_train_path = "zh_data/ore_train_zh.json"
oie_dev_path = "zh_data/oie_dev_zh.json"
ore_dev_path = "zh_data/ore_dev_zh.json"
oie_test_path = "zh_data/oie_test_zh.json"
ore_test_path = "zh_data/ore_test_zh.json"

with open(oie_train_path) as f:
    oie_train = json.load(f)
with open(ore_train_path) as f:
    ore_train = json.load(f)
with open(oie_dev_path) as f:
    oie_dev = json.load(f)
with open(ore_dev_path) as f:
    ore_dev = json.load(f)
with open(oie_test_path) as f:
    oie_test = json.load(f)
with open(ore_test_path) as f:
    ore_test = json.load(f)

with open("zh_data/dp_emb_train_zh.json", 'w') as wf:
    dp_train = oie_train+ore_train+oie_dev+ore_dev
    json.dump(dp_train, wf, ensure_ascii=False)
with open("zh_data/dp_emb_dev_zh.json", 'w') as wf:
    dp_test = oie_test+ore_test
    json.dump(dp_test, wf, ensure_ascii=False)


oie_train_path = "zh_data/oie_train_matrixs_zh.npy"
ore_train_path = "zh_data/ore_train_matrixs_zh.npy"
oie_dev_path = "zh_data/oie_dev_matrixs_zh.npy"
ore_dev_path = "zh_data/ore_dev_matrixs_zh.npy"
oie_test_path = "zh_data/oie_test_matrixs_zh.npy"
ore_test_path = "zh_data/ore_test_matrixs_zh.npy"

oie_train = np.load(oie_train_path, allow_pickle=True)
ore_train = np.load(ore_train_path, allow_pickle=True)
oie_dev = np.load(oie_dev_path, allow_pickle=True)
ore_dev = np.load(ore_dev_path, allow_pickle=True)
oie_test = np.load(oie_test_path, allow_pickle=True)
ore_test = np.load(ore_test_path, allow_pickle=True)

dp_train = np.concatenate([oie_train, ore_train, oie_dev, ore_dev], axis=0)
np.save("zh_data/dp_emb_train_matrixs_zh.npy", dp_train)
dp_test = np.concatenate([oie_test, ore_test], axis=0)
np.save("zh_data/dp_emb_dev_matrixs_zh.npy", dp_test)
