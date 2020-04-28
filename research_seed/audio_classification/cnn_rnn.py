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


class LSTMBlock(nn.Module):
    def __init__(self, input_size=300, hidden_size=256, num_layers=2, bidirectional=True, dropout=0.0, num_classes=8):
        super(LSTMBlock, self).__init__()

        self.input_size = input_size
        self.num_layers = num_layers   # RNN hidden layers
        self.hidden_size = hidden_size                 # RNN hidden nodes
        self.num_classes = num_classes
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1  # 1 for bidirectional lstm

        self.LSTM = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            # input & output  has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            batch_first=True,
        )

        self.fc1 = nn.Linear(self.hidden_size * self.num_directions, self.num_classes)

    def forward(self, x):

        self.LSTM.flatten_parameters()
        # print(x.shape)

        RNN_out, (h_n, h_c) = self.LSTM(x, None)
        # out" will give you access to all hidden states in the sequence
        """ h_n shape ((num_layers * num_directions, batch, hidden_size)), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        #print(RNN_out.size())
        # print(h_n.size())
        #x = h_n.view(x.shape[0], self.num_layers, self.num_directions, self.hidden_size)
        # print(x.shape) #batch, num_layers, num_directions, hidden_size
        #x = x[:, self.num_layers - 1, self.num_directions - 1]
        # print(x.shape)
        x = self.fc1(RNN_out[:,-1,:])   # choose RNN_out at the last time step and activations in both directions
        #print(x.shape)
        # print(x.shape)
        return x


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


class CNN_RNN(pl.LightningModule):

    def __init__(self, hparams):
        super(CNN_RNN, self).__init__()

        self.hparams = hparams
        self.num_classes = hparams.num_classes
        self.bidirectional = bool(hparams.bidirectional)
        self.num_layers_rnn = hparams.num_layers_rnn
        self.dropout_rnn = hparams.dropout_rnn
        self.num_layers_rnn = hparams.num_layers_rnn   # RNN hidden layers
        self.hidden_size_rnn = hparams.hidden_size_rnn                 # RNN hidden nodes

        self.lflb1 = LFLBlock(inp_ch=1, out_ch=64, conv_k=3,
                              conv_s=1, pool_k=2, pool_s=2, p_dropout=self.hparams.dropout_1)
        self.lflb2 = LFLBlock(inp_ch=64, out_ch=64, conv_k=3,
                              conv_s=1, pool_k=4, pool_s=4, p_dropout=self.hparams.dropout_2)
        self.lflb3 = LFLBlock(inp_ch=64, out_ch=128, conv_k=3,
                              conv_s=1, pool_k=4, pool_s=4, p_dropout=self.hparams.dropout_3)
        self.lflb4 = LFLBlock(inp_ch=128, out_ch=128, conv_k=3,
                              conv_s=1, pool_k=4, pool_s=4, p_dropout=self.hparams.dropout_3)

        self.rnn = LSTMBlock(input_size=128, hidden_size=self.hidden_size_rnn,dropout=self.dropout_rnn, num_classes=self.num_classes,
                             bidirectional=self.bidirectional, num_layers=self.num_layers_rnn,
                             )
        # if self.hparams.rnn:
        #
        # else:
        #     self.lflb4 = LFLBlock(128,128,3,2)
        #     self.dropout4 = nn.Dropout2d(p=0.1)

        #     self.fc1 = nn.Linear(128 * 1, 64)
        #     self.fc2 = nn.Linear(64, 8)

    def forward(self, x):
        #print('is cuda', x.is_cuda)
        x = self.lflb1(x)

        # print(x.shape) #torch.Size([32, 64, 144, 109])
        x = self.lflb2(x)

        # print(x.shape) #torch.Size([32, 64, 70, 52]])

        x = self.lflb3(x)  # torch.Size([[32, 128, 22, 24])
        # print(x.shape)

        x = self.lflb4(x)  # torch.Size([32, 64, 9, 10])
        # print(x.shape)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.shape[0], x.shape[1], -1)

        # print(x.shape)
        x = self.rnn(x)
        
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
        #self.logger.experiment.add_scalar('Loss/train', loss_val, self.global_step)
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def accuracy(self, y_true, y_pred):
        with torch.no_grad():
            acc = (y_true == y_pred).sum().to(torch.float32)
            acc /= y_pred.shape[0]

            return acc

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch

        y_hat = self.forward(x)

        with torch.no_grad():
            y_pred = torch.max(F.softmax(y_hat, dim=1), 1)[1]
            #print(y.cpu(), y_pred.cpu())
            acc = metrics.accuracy_score(y.cpu(), y_pred.cpu())
            f1 = metrics.f1_score(y.cpu(), y_pred.cpu(), average='macro')
        loss_val = F.cross_entropy(y_hat, y)

        output = OrderedDict(
            {'val_loss': loss_val, 'val_f1': f1, 'val_acc': acc})

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

        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict,
                  'val_loss': tqdm_dict["val_loss"]}
        # self.logger.experiment.flush()
        return result

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers

        # ,weight_decay=0.01)#torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return AdaBound(self.parameters(), lr=self.hparams.learning_rate_init,
                        final_lr=self.hparams.learning_rate_final, weight_decay=self.hparams.weight_decay)

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

        parser.add_argument('--learning_rate_init',
                            default=0.0002898, type=float)
        parser.add_argument('--learning_rate_final',
                            default=0.01435, type=float)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--weight_decay', default=0.004566, type=float)
        #cnn
        parser.add_argument('--dropout_1', default=0.5424, type=float)
        parser.add_argument('--dropout_2', default=0.257, type=float)
        parser.add_argument('--dropout_3', default=0.558, type=float)
        #rnn
        parser.add_argument('--bidirectional', default=1, type=int)
        parser.add_argument('--num_layers_rnn', default=2, type=int)
        parser.add_argument('--dropout_rnn', default=0.0, type=float)
        parser.add_argument('--hidden_size_rnn', default=256, type=int)



        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=10000, type=int)

        # data
        parser.add_argument(
            '--data_root', default='../datasets/RAVDESS/SOUND_SPECT/', type=str)
        parser.add_argument(
            '--num_classes', dest='num_classes', default=8, type=int)
        return parser

# %%
# #os.chdir('../../')
# modl = CNN_RNN('')
# #%%
# inpt = torch.Tensor(3,3,94,124)
# #inpt.permute(0,3,1,2).shape
# #inpt = inpt.view(3,10,-1)
# #inpt.shape
# #%%
# modl(inpt).shape
# #%%
# from torch.nn.utils.rnn import pack_sequence
# a = torch.tensor([1,2,3])
# b = torch.tensor([4,5])
# c = torch.tensor([6])
# pack_sequence([a, b, c])[0]
# #%%
# o = rnnout.view(1,10,1,256)
# h = hid[0].view(1,2,1,256)
# # %%
# o[0].shape
# h[0].shape
# h[0][1] == o[0][9]

# # %%
# print(inpt.shape)
# inpt[:,:,:].shape
