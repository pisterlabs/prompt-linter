# -*- coding: utf-8 -*-
"""
Training of simple model for FINCO using PyTorch

Should be run from outside the package
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from finco import load_results
from finco.ml.torch.finco_dataset import FINCODataset
from finco.ml.torch.coherent_loss import CoherentLoss

from splitting_method import SplittingMethod

#%% Preprocessing
resfile = ''
result = load_results('trajs_1_T_2.0_dt_0.02.hdf')

a = 0.5
b = 0.1
chi = 2j
gamma0 = 0.5

def psi0(x):
    return (2*gamma0/np.pi)**0.25 * np.exp(-gamma0 * (x-np.conj(chi)/2/gamma0)**2-(chi.imag)**2/4/gamma0)

def H_p(x):
    return a*x**2 + b*x**4

def H_k(p):
    return p ** 2 / 2

T = 2
dt = T / 100
spl = SplittingMethod(x0 = -50, x1 = 50, dx = 1e-2, 
                      T = T, dt = dt, trecord = dt, imag = False,
                      psi0 = psi0, H_p = H_p, H_k = H_k) 


spl.propagate()

#%% Train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device {device}')
lr = 1e-3
epochs = 250

loader = DataLoader(FINCODataset(result, -5, 5, 100, -5, 5, 100, 100), batch_size=1, shuffle=True, drop_last=True)
n_trajs = loader.dataset.n_trajs
model = torch.nn.Sequential(torch.nn.Flatten(),
                            torch.nn.Linear(11*2*n_trajs,n_trajs,bias=False),
                            torch.nn.Sigmoid())
loss = CoherentLoss(-5, 5, 1000, -5, 5, 100, spl)
optim = torch.optim.SGD(model.parameters(), lr=lr)

model.to(device=device)
loss.to(device=device)
for epoch in range(epochs):
    print(f'Epoch {epoch+1} -----------------------')
    for batch, x in enumerate(loader):
        x = x.to(device=device)
        model.zero_grad()
        y = model(x)
        l = loss(y, x)
        l.backward()
        optim.step()
        if (batch+1) % 10 == 0:
            print(f'batch {batch+1:>5d}: loss {l.item():>7f}')
