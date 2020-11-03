"""
training process entry
"""
import json
import os
import time
import torch
import numpy as np
from torch import mode

from transformers import BertTokenizer, BertConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
from models.main_model import TransModel
from data_reader import OIEDataset
from config import FLAGS, aggcnargs


def select_model(flags, bertconfig, aggcnargs):
    model = TransModel(flags, bertconfig, aggcnargs)
    return model


def select_optim(flags, model):
    optimizer = torch.optim.Adadelta([{"params": model.aggcn.parameters(), "lr": aggcnargs.lr},
                                      {"params": model.bert.parameters()},
                                      {"params": model.transd.parameters()}], lr=FLAGS.learning_rate)
    return optimizer


def main():
    # load from pretrained config file
    bertconfig = json.load(open(FLAGS.pretrained_config))
    bertconfig = BertConfig(**bertconfig)

    # Initiate model
    print("Initiating model.")
    model = select_model(FLAGS, bertconfig, aggcnargs)
    if torch.cuda.is_available():
        model.cuda()
    if FLAGS.is_continue:
        model.load_state_dict(torch.load(FLAGS.dp_checkpoint_path))
        print('Training from previous model.')
    print("Model initialized.")

    # load data
    print("Loading traning and valid data")
    tokenizer = BertTokenizer.from_pretrained(FLAGS.pretrained_vocab)
    train_set = OIEDataset(FLAGS.train_path, FLAGS.train_mat, tokenizer, FLAGS.max_length, mode="train")
    dev_set = OIEDataset(FLAGS.dev_path, FLAGS.dev_mat, tokenizer, FLAGS.max_length, mode="train")
    trainset_loader = torch.utils.data.DataLoader(train_set, FLAGS.batch_size, shuffle=False, num_workers=0, drop_last=True)
    validset_loader = torch.utils.data.DataLoader(dev_set, FLAGS.test_batch_size, shuffle=False, num_workers=0, drop_last=True)

    optimizer = select_optim(FLAGS, model)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=1, patience=3, factor=0.5, min_lr=1.e-8)
    # scheduler = CosineAnnealingLR(optimizer, T_max=(FLAGS.epoch // 9) + 1)
    best_loss = 1000
    patience = 6

    print("Start training", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    for epoch in range(FLAGS.epoch):
        model.train()
        losses = []
        with tqdm(total=len(train_set)/FLAGS.batch_size, desc=f'Epoch {epoch+1}/{FLAGS.epoch}', unit='it') as pbar:
            for step, data in enumerate(trainset_loader):
                token, pos, ner, arc, matrix, gold, mask, acc_mask = data
                model.zero_grad()
                loss = model(token, pos, ner, arc, matrix, gold, mask, acc_mask)
                losses.append(loss.data.item())
                # backward
                loss.backward()
                optimizer.step()
                # tqdm
                pbar.set_postfix({'batch_loss': loss.data.item()})   # 在进度条后显示当前batch的损失
                pbar.update(1)
        train_loss = np.mean(losses)
        print(f"[{epoch + 1}/{FLAGS.epoch}] trainset mean_loss: {train_loss: 0.4f}")

        model.eval()
        losses = []
        for step, data in enumerate(validset_loader):
            token, pos, ner, arc, matrix, gold, mask, acc_mask = data
            model.zero_grad()
            loss = model(token, pos, ner, arc, matrix, gold, mask, acc_mask)
            losses.append(loss.data.item())
        valid_loss = np.mean(losses)
        scheduler.step(valid_loss)
        # scheduler.step(epoch)
        print(f"[{epoch + 1}/{FLAGS.epoch}] validset mean_loss: {valid_loss: 0.4f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            print('Saving model...  ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            torch.save(model.state_dict(), FLAGS.checkpoint_path)
            model.save_embedding()
            print('Saving model finished.  ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            patience = 6
        else:
            patience -= 1

        if patience == 0:
            break
    print('Training finished.  ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


if __name__ == "__main__":
    main()
