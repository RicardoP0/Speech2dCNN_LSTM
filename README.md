### pytorch-lightning-conference-seed
Use this seed to refactor your PyTorch research code for:  
- a paper submission  
- a new research project.     

[Read the usage instructions here](https://github.com/williamFalcon/pytorch-lightning-conference-seed/blob/master/HOWTO.md)

#### Goals  
The goal of this seed is to structure ML paper-code the same so that work can easily be extended and replicated.   

###### DELETE EVERYTHING ABOVE FOR YOUR PROJECT   
---   
<div align="center">    
 
# Speech2dCNN_LSTM    



<!--  
Conference   
-->   
</div>
 
## Description   
A pytorch implementation of [Speech emotion recognition using deep 1D & 2D CNN LSTM networks](https://www.sciencedirect.com/science/article/abs/pii/S1746809418302337) using pytorch lighting and wandb sweep for hyperparameter finding. I'm not affiliated with the authors of the paper.

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
 Next, navigate to [Your Main Contribution (MNIST here)] and run it.   
 ```bash
# module folder
cd research_seed/mnist/   

# run module (example: mnist as your main contribution)   
python mnist_trainer.py    
```

## Main Contribution      
List your modules here. Each module contains all code for a full system including how to run instructions.   
- [MNIST](https://github.com/williamFalcon/pytorch-lightning-conference-seed/tree/master/research_seed/mnist)  


