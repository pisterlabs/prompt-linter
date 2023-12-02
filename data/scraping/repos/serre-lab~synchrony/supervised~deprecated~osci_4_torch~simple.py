import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
import torch
from torch.autograd import Variable
import numpy as np
import os
from tqdm import tqdm
import ipdb
import pandas as pd

def potential(k,phase):
    '''Potential energy which is minimized during dynamics.'''
    return -1* k * torch.cos(phase[0] - phase[1])

def coherence(phase):
    '''Loss function optimized to learn coupling.'''
    return .5*(torch.sqrt(torch.cos(phase).sum()**2 + torch.sin(phase).sum()**2))

def inner_opt(k, num_steps=20, lr=.1):
   ''' Optimization loop for dynamics. '''
   phase = Variable(2*np.pi*torch.rand(2,), requires_grad=True)
   inner_optim = torch.optim.SGD((phase,), lr=lr)  

   coherence_history  = []
   energy_history = []
   for n in range(num_steps):
       inner_optim.zero_grad()
       energy = potential(k, phase)
       coh = coherence(phase)
       energy_history.append(energy.data.numpy())
       coherence_history.append(coh.data.numpy())
       energy.backward()
       inner_optim.step()
   return phase, energy_history, coherence_history

def outer_opt(maximize=False, k_mag=1.0, num_steps=1000,lr=1e-4, inner_show=25, outer_show=25):
    ''' Optimization loop for coupling'''
    sign = -1.0 if maximize else 1.0
    k = Variable(torch.tensor(sign*k_mag), requires_grad=True)
    outer_optim = torch.optim.SGD((k,),lr=lr)
    k_history    = []
    loss_history = []
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for m in tqdm(range(num_steps)):
        outer_optim.zero_grad()
        min_phase, energy_history, coherence_history = inner_opt(k)
        loss      = sign*coherence(min_phase)
        loss_history.append(loss.data.numpy())
        k_history.append(k.clone().data.numpy())
        loss.backward()
        outer_optim.step()
        if m % outer_show ==0:
            for array, name in zip([energy_history, loss_history, k_history], ['potential', 'coherence', 'coupling']):
                if name == 'coherence' and maximize==True:
                    array = -1*np.array(array) 
                plt.plot(array) 
                plt.title(name)
                plt.savefig(os.path.join(save_dir, name + '.png'))
                plt.close()

if __name__=='__main__':
    '''This experiment tries to switch between coherence and incoherence of two oscillators using gradient descent. 
        * To switch from coherence to incoherence: maximize=True
        * To switch from incorherence to coherence: maximize=False
        k_mag controls the magnitude of the initial coupling. 
        When maximize is true, the sign of the initial coupling is negative. '''

    save_dir = os.path.join(os.path.expanduser('~'), 'oscillators')
    display_true = True
    num_steps = 100
    num_k = 1000
    num_d = 4
    averaging_window_size = 50
    if display_true:
        coh = []
        time_averaged = []
        ks = np.linspace(-1,1,num_k)
        for k in tqdm(ks):
            phase, energy_history, coherence_history = inner_opt(k, num_steps=num_steps)
            coh.append(coherence(phase).data.numpy())  
            ta = np.array([1./num_steps * (np.arange(num_steps)**d * np.array(coherence_history)).sum() for d in range(num_d)])
            time_averaged.append(ta)

        coh = np.array(coh)
        coh = np.convolve(coh, np.ones(averaging_window_size), mode='same')
        coh = np.expand_dims(coh, axis=1) / coh.max()

        time_averaged = np.array(time_averaged)
        time_averaged = np.array([np.convolve(ta, np.ones(averaging_window_size), mode='same') for ta in time_averaged.T]).T
        time_averaged /= np.max(time_averaged, axis=0)

        all_coherence = np.concatenate((coh, time_averaged), axis=1)
        rolling_stds = np.array([pd.rolling_std(coh, averaging_window_size) for coh in all_coherence.T]).T
        plt.plot(ks[:-1*averaging_window_size - 1], all_coherence[:-1*averaging_window_size - 1,:], alpha=.5)
        for i in range(num_d + 1):
            plt.fill_between(ks[:-1*averaging_window_size - 1], all_coherence[:-1*averaging_window_size - 1,i] - rolling_stds[:-1*averaging_window_size - 1,i], all_coherence[:-1*averaging_window_size - 1,i] + rolling_stds[:-1*averaging_window_size - 1,i], alpha=.2)
        plt.legend(('Convergent', 'Average', 'Monic', 'Quadratic', 'Cubic'))
        plt.xlabel('K')
        plt.ylabel('Time averaged coherence')
        plt.savefig(os.path.join(save_dir, 'coupling_vs_ta.png'))
        plt.close()
    #outer_opt(maximize=True, k_mag = 1.0, lr=1e-4)
