"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
# %%
import os
import sys
current_path = os.path.abspath('.')
sys.path.append(current_path)

import logging
from shutil import copyfile
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import torch.nn.functional as F
import torch
from research_seed.audio_classification.datasets.iemocap_spect import IEMOCAPSpectDataset
from pytorch_lightning.callbacks import EarlyStopping
import wandb
from pytorch_lightning.loggers import WandbLogger
from research_seed.audio_classification.cnn_rnn import CNN_RNN
from research_seed.audio_classification.cnn_lflb import CNN_LFLB

from argparse import ArgumentParser
from pytorch_lightning import Trainer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


def report(model, wandb_logger):
    # https://donatstudios.com/CsvToMarkdownTable

    model.eval()
    model = model.cpu()
    y_pred = []
    y_true = []

    for x, y in model.val_dataloader():

        res = torch.max(F.softmax(model(x), dim=1), 1)[1].numpy()
        y_pred.extend(res)
        y_true.extend(y.numpy())

    unique_label = np.unique([y_true, y_pred])
    cmtx = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=unique_label),
        index=['true:{:}'.format(x) for x in unique_label],
        columns=['pred:{:}'.format(x) for x in unique_label]
    )

    report = pd.DataFrame(classification_report(
        y_true, y_pred, output_dict=True))
    print(cmtx, report)
    wreport = []
    tmp = [str(item) for item in report.values[0]]
    tmp.insert(0, 'precision')
    wreport.append(tmp)
    tmp = [str(item) for item in report.values[1]]
    tmp.insert(0, 'recall')
    wreport.append(tmp)
    tmp = [str(item) for item in report.values[2]]
    tmp.insert(0, 'f1-score')
    wreport.append(tmp)
    tmp = [str(item) for item in report.values[3]]
    tmp.insert(0, 'support')
    wreport.append(tmp)

    hreport = report.columns
    hreport = hreport.insert(0, '')

    wandb_logger.log_metrics({'confusion_matrix': wandb.plots.HeatMap(unique_label, unique_label, cmtx.values, show_text=True),
                              'classification_report': wandb.Table(data=wreport, columns=hreport.values)})


def main(hparams, network):
    # init module

    model = network(hparams)
    project_folder = 'audio_emotion_team'
    wandb_logger = WandbLogger(
        name='lflb_dropout_rnn', project=project_folder, entity='thesis', offline=False)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=False,
        mode='min'
    )

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
        logger=wandb_logger,
        weights_summary='full',
        early_stop_callback=early_stop_callback,
        profiler=True,
        benchmark=True,
        log_gpu_memory='all'

    )
   
    wandb_logger.experiment.config.update(
        {'dataset': 'IEMOCAP_SPECT_GS_8s_512h_2048n'})
    wandb_logger.watch(model)

    trainer.fit(model)
    # load best model
    exp_folder = project_folder + '/version_'+wandb_logger.experiment.id
    model_file = os.listdir(exp_folder + '/checkpoints')[0]
    # eval and upload best model
    model = network.load_from_checkpoint(
        exp_folder+'/checkpoints/' + model_file)
    report(model, wandb_logger)
    copyfile(exp_folder+'/checkpoints/' + model_file,
             wandb_logger.experiment.dir+'/model.ckpt')
    wandb_logger.experiment.save('model.ckpt')


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str, default=1)
    parser.add_argument('--nodes', type=int, default=1)

    network = CNN_RNN#CNN_LFLB  # CNN_RNN
    parser = network.add_model_specific_args(parser)

    # parse params
    print(os.getcwd())
    hparams = parser.parse_args(["--data_root", "../datasets/IEMOCAP/SOUND_SPECT_GS_8s_512h_2048n/", '--max_nb_epochs', '10000',
                                 '--num_classes', '8'])

    main(hparams, network)


# %%
