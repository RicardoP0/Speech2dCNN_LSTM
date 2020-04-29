   
<div align="center">    
 
# Speech2dCNN_LSTM    



<!--  
Conference   
-->   
</div>
 
## Description   
A pytorch implementation of [Speech emotion recognition using deep 1D & 2D CNN LSTM networks](https://www.sciencedirect.com/science/article/abs/pii/S1746809418302337) using pytorch lighting and wandb sweep for hyperparameter finding. I'm not affiliated with the authors of the paper.

![Example of spectogram image used as input](/img/spectogram.png)
## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/RicardoP0/Speech2dCNN_LSTM.git

# install project   
cd Speech2dCNN_LSTM
pip install -e .   
pip install requirements.txt
 ```   
 Next, navigate to [CNN+LSTM](https://github.com/RicardoP0/Speech2dCNN_LSTM/tree/master/research_seed/audio_classification)   and run it.   
 ```bash
# module folder
cd research_seed/audio_classification/   

# run module
python cnn_trainer.py    
```

## Main Contribution      
 
- [CNN+LSTM](https://github.com/RicardoP0/Speech2dCNN_LSTM/tree/master/research_seed/audio_classification)  

## Results

Validation accuracy reaches 0.4 and a F1 value of 0.3 using 8 classes.
![Validation accuracy on 8 classes](/img/val_acc.png)

