import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import int_data as syn
import torch
import torch.nn as nn
import torch.nn.init as init

# import neurogym as ngym
device = torch.device('cuda')

class MPN(nn.Module):
    def __init__(self, dims):
        super(MPN, self).__init__()

        # input size, hidden size, output size respectively
        Nx, Nh, Ny = dims

        # MPN layer
        self.eta = torch.empty(1, 1)
        self.lam = torch.empty(1, 1)
        self.Wi = torch.empty(Nx, Nh)
        self.M  = torch.zeros(Nx, Nh)
        self.a1 = nn.Tanh()

        self.linear = nn.Linear(Nh, Ny)
        self.a2 = nn.Tanh()

        init.xavier_uniform_(self.Wi)
        init.xavier_uniform_(self.eta)
        init.xavier_uniform_(self.lam)

    # we need reset_state to have batch size cause there is a different M_t for each batch
    # def reset_state(self, batchSize):
    #     self.M = torch.ones(batchSize, *self.w1.shape, device=self.w1.device) #shape=[B,Ny,Nx]   
    #     self.M = self.M * self.M0.unsqueeze(0) # (B, Ny, Nx) x (1, Ny, Nx)

    def update_sm(self, x, h):
        # c * (Nx, Nh) + c * (Nx, B) @ (B, Nh) = (Nx, Nh)
        self.M = self.lam * self.M + self.eta * x.T @ h

    def forward(self, x):
        # (B, Nx) @ (Nx, Nh) + (B, Nx) @ (Nx, Nh) = (B, Nh)
        h = self.a1(x @ (self.Wi * self.M) + x @ self.Wi)
        self.update_sm(x, h)

        # (B, Nh) @ (Nh, Ny) + (1,Ny) = (B, Ny)
        return self.a2(self.linear(h))
    
    # evaluate network on sequence of inputs
    # THIS IS WRONG im still trying to figure out how batches work with this
    def evaluate(self, x):
        y = torch.zeros_like(x, dtype=torch.float32)

        for i in range(x.size(1)):
            y[i] = self.forward(x[i])

        return y
    
    def getM(self):
        return self.M
    
net = MPN((2,100,2))
data = torch.tensor([[0.5,1.0],[1.0,1.0],[2.0,2.0]])
print(data.shape)
net.forward(data)
print(net.getM())



########## Toy data parameters ##########
toy_params = {
    'data_type': 'int', 
    'dataset_size': 3200,
    'phrase_length': 20,
    'n_classes': 3,
    'input_type': 'binary',    # one_hot, binary, binary1-1
    'input_size': 50,          # defaults to length of words
    'include_eos': True,

    'n_delay': 0, # Inserts delay words (>0: at end, <0: at beginning)

    'uniform_score': True, # Uniform distribution over scores=
}


# TEST DATAS
trainData, trainOutputMask, toy_params = syn.generate_data(
    toy_params['dataset_size'], toy_params, toy_params['n_classes'], 
    verbose=False, auto_balance=False, device=device)

dataLoader = DataLoader(trainData, batch_size = 10, shuffle=True)

i = 0
for X, Y in dataLoader:
    if i == 0:
        print(X.shape, Y.shape)
        print(X[0])
        print(Y[0])
    
    i += 1
