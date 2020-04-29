"""
This file defines the core research contribution   
"""
# %%
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from argparse import ArgumentParser
from collections import OrderedDict
from research_seed.audio_classification.datasets.iemocap_spect import IEMOCAPSpectDataset
import pytorch_lightning as pl
import numpy as np
import sklearn.metrics as metrics
from adabound import AdaBound

import torchvision.models as models


class LFLBlock(nn.Module):
    def __init__(self, inp_ch, out_ch, conv_k, conv_s, pool_k, pool_s, p_dropout):
        super(LFLBlock, self).__init__()

        self.conv = nn.Conv2d(inp_ch, out_ch, conv_k, conv_s, padding=(1, 2))
        self.batch_nm = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool_k, pool_s)

        self.dropout = nn.Dropout2d(p=p_dropout)  # AlphaDropout
        self.actv = nn.ELU()

    def forward(self, x):

        x = self.conv(x)

        x = self.actv(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.batch_nm(x)

        return x


class CNN_LFLB(pl.LightningModule):

    def __init__(self, hparams):
        super(CNN_LFLB, self).__init__()
        # not the best model...

        self.hparams = hparams
        self.num_classes = hparams.num_classes

        self.lflb1 = LFLBlock(inp_ch=1, out_ch=64, conv_k=3,
                              conv_s=1, pool_k=2, pool_s=2, p_dropout=self.hparams.dropout_1)
        self.lflb2 = LFLBlock(inp_ch=64, out_ch=64, conv_k=3,
                              conv_s=1, pool_k=4, pool_s=4, p_dropout=self.hparams.dropout_2)
        self.lflb3 = LFLBlock(inp_ch=64, out_ch=128, conv_k=3,
                              conv_s=1, pool_k=4, pool_s=4, p_dropout=self.hparams.dropout_3)
        self.lflb4 = LFLBlock(inp_ch=128, out_ch=128, conv_k=3,
                              conv_s=1, pool_k=4, pool_s=4, p_dropout=self.hparams.dropout_3)

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = self.lflb1(x)
        x = self.lflb2(x)
        x = self.lflb3(x) 
        x = self.lflb4(x)

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss_val = F.cross_entropy(y_hat, y)
        with torch.no_grad():
            y_pred = torch.max(F.softmax(y_hat, dim=1), 1)[1]
            acc = metrics.accuracy_score(y.cpu(), y_pred.cpu())
        tqdm_dict = {'train_loss': loss_val, 'train_acc': acc}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch

        with torch.no_grad():
            y_hat = self.forward(x)
            y_pred = torch.max(F.softmax(y_hat, dim=1), 1)[1]

            acc = metrics.accuracy_score(y.cpu(), y_pred.cpu())
            f1 = metrics.f1_score(y.cpu(), y_pred.cpu(), average='macro')
            loss_val = F.cross_entropy(y_hat, y)

        output = OrderedDict({'val_loss': loss_val, 'val_f1': f1, 'val_acc': acc})
     
        return output

    def validation_end(self, outputs):
        # OPTIONAL
        tqdm_dict = {}

        for metric_name in ["val_loss", "val_f1", "val_acc"]:
            metric_total = 0

            for output in outputs:
                metric_value = output[metric_name]

                # reduce manually when using dp
                if self.trainer.use_dp or self.trainer.use_ddp2:
                    metric_value = torch.mean(metric_value)

                metric_total += metric_value

            tqdm_dict[metric_name] = metric_total / len(outputs)

        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict,'val_loss': tqdm_dict["val_loss"]}
 
        return result

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers

        # ,weight_decay=0.01)#torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return AdaBound(self.parameters(), lr=self.hparams.learning_rate_init, 
        final_lr=self.hparams.learning_rate_final,weight_decay=self.hparams.weight_decay)

    def train_dataloader(self):
        # REQUIRED
        transform = transforms.Compose([transforms.ToTensor()])
        return DataLoader(IEMOCAPSpectDataset(self.hparams.data_root, set_type='train', transform=transform, num_classes=self.num_classes),
                          batch_size=32, num_workers=2, pin_memory=True,
                          shuffle=True)

    def val_dataloader(self):
        # OPTIONAL

        transform = transforms.Compose([transforms.ToTensor()])
        return DataLoader(IEMOCAPSpectDataset(self.hparams.data_root, set_type='val', transform=transform, num_classes=self.num_classes),
                          batch_size=32, num_workers=2, pin_memory=True,
                          shuffle=True)

    def test_dataloader(self):
        # OPTIONAL
        transform = transforms.Compose([transforms.ToTensor()])
        return DataLoader(IEMOCAPSpectDataset(self.hparams.data_root, set_type='test', transform=transform, num_classes=self.num_classes),
                          batch_size=32, num_workers=2, pin_memory=True,
                          shuffle=True)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate_init', default=7e-4, type=float)
        parser.add_argument('--learning_rate_final', default=0.02, type=float)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--dropout_1', default=0.6, type=float)
        parser.add_argument('--dropout_2', default=0.3, type=float)
        parser.add_argument('--dropout_3', default=0.2, type=float)
        parser.add_argument('--weight_decay', default=0.0, type=float)


        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=10000, type=int)

        # data
        parser.add_argument(
            '--data_root', default='../datasets/RAVDESS/SOUND_SPECT/', type=str)
        parser.add_argument(
            '--num_classes', dest='num_classes', default=8, type=int)
        return parser
