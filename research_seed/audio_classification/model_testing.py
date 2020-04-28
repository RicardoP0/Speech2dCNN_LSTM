"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
#%%
import os
import sys
current_path = os.path.abspath('.')
sys.path.append(current_path)
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import Profiler
from argparse import ArgumentParser

from research_seed.audio_classification.cnn_spect import CNN_SPECT
from research_seed.audio_classification.cnn_lflb import CNN_LFLB
from research_seed.audio_classification.cnn_rnn import CNN_RNN
from pytorch_lightning.loggers import WandbLogger
import wandb
from pytorch_lightning.callbacks import EarlyStopping
from research_seed.audio_classification.datasets.iemocap_spect import IEMOCAPSpectDataset

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np

from shutil import copyfile
import sys
import logging
# ...
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)
def report(model, wandb_logger):
    #https://donatstudios.com/CsvToMarkdownTable

    model.eval()
    model = model.cpu()
    y_pred = []
    y_true = []
   
    for x,y in model.val_dataloader():
       
        res = torch.max(F.softmax(model(x), dim=1),1)[1].numpy()
        y_pred.extend(res)
        y_true.extend(y.numpy())
        break
        
        

    unique_label = np.unique([y_true, y_pred])
    cmtx = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=unique_label), 
        index=['true:{:}'.format(x) for x in unique_label], 
        columns=['pred:{:}'.format(x) for x in unique_label]
    )
  
    report = pd.DataFrame(classification_report(y_true,y_pred, output_dict=True))
        
    wreport = []
    tmp = [str(item) for item in report.values[0]]
    tmp.insert(0,'precision')
    wreport.append(tmp)
    tmp = [str(item) for item in report.values[1]]
    tmp.insert(0,'recall')
    wreport.append(tmp)
    tmp = [str(item) for item in report.values[2]]
    tmp.insert(0,'f1-score')
    wreport.append(tmp)
    tmp = [str(item) for item in report.values[3]]
    tmp.insert(0,'support')
    wreport.append(tmp)

    print(report,cmtx)



    hreport = report.columns
    hreport = hreport.insert(0,'')

    if wandb_logger:
        wandb_logger.log_metrics({'confusion_matrix': wandb.plots.HeatMap(unique_label, unique_label, cmtx.values, show_text=True),
        'classification_report':wandb.Table(data=wreport, columns=hreport.values)})
def main(hparams, network):
    # init module
    debugging = True

    project_folder = 'test'
    model = network(hparams)
    #wandb_logger = WandbLogger(name='test',offline=True,project=project_folder,entity='ricardop0')
    
   
    early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=5,
    verbose=False,
    mode='min'
    )
 
    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,   
        fast_dev_run=debugging,
        weights_summary='full',
        early_stop_callback=early_stop_callback,
        profiler=True,
        benchmark=True,
        log_gpu_memory='all',
        overfit_pct = 0.1,
        #logger=wandb_logger,

    )
    
    trainer.fit(model)
    # id = wandb_logger.experiment.id
    # print(id)
    # os.environ["WANDB_RUN_ID"] = id
    # wandb_logger = WandbLogger(name='test',offline=True,project=project_folder,entity='ricardop0')
    # wandb_logger.experiment
    # print(wandb_logger.experiment.id)
    #     # load best model
    # exp_folder = project_folder + '/version_'+wandb_logger.experiment.id
    # model_file = os.listdir(exp_folder + '/checkpoints')[0]
    # # eval and upload best model
    # model = network.load_from_checkpoint(
    #     exp_folder+'/checkpoints/' + model_file)
    # report(model, wandb_logger)
    # copyfile(exp_folder+'/checkpoints/' + model_file,
    #          wandb_logger.experiment.dir+'/model.ckpt')
    # wandb_logger.experiment.save('model.ckpt')
    # print('here')
    # wandb_logger.finalize()
   
    #print(wandb_logger.experiment.config)
    #print(wandb_logger.experiment.id)
    #exp_folder = 'audio_class/version_'+wandb_logger.experiment.id
    #model_file = os.listdir(exp_folder + '/checkpoints')[0]
    #model = CNN_LFLB.load_from_checkpoint(exp_folder+'/checkpoints/'+ model_file)
    #report(model, None)
    #wandb_logger.experiment.save(exp_folder+'/checkpoints/'+ model_file)



if __name__ == '__main__':
    
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str, default=1)
    parser.add_argument('--nodes', type=int, default=1)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    network = CNN_RNN#CNN_LFLB#CNN_RNN
    parser = network.add_model_specific_args(parser)

    # parse params
    print(os.getcwd())
    hparams = parser.parse_args(["--data_root", "../datasets/IEMOCAP/SOUND_SPECT_GS_8s_512h_2048n/", '--max_nb_epochs', '50',
                                 '--num_classes', '6'])
 
    main(hparams, network)

   

#%%

