# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 08:27:58 2022

@author: bene-
"""

#!/usr/bin/env python
# coding: utf-8

import numpy as np
from  math import *
import matplotlib.pyplot as plt
import scipy.optimize as so
import scipy.special as sp
import mpmath as mp

import os
import openai

openai.api_key = "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"




# The graph looks like we expected it. It shows that for large $N_{md}$ the energy of the old and new configuration is almost the same. This will later be important to achieve a high acceptance probability.  

# In[35]:

N_md = 10#Leapfrog integration steps
N_en = 10
N_cfg = 100000
beta=1000
N=20
h=0.5
beta_h=0.5
J=1/N
phi=np.array((800,800,600))
p_0 = 0 
I = 10
A = 1 
ar=0
f_i=np.array((0.96,1.025,1.055,1.085,1.13))*1000
x=np.array((0.176,0.234,0.260,0.284,0.324))
delta_f=np.array((0.025,0.02,0.015,0.01,0.008))*1000


# This is just defining of variables. 

# In[36]:


def leapfrog_plot():
    global N_md
    p=np.zeros(3)
    phi=np.array((00,00,00))
    for i in range(100):
        for j in range(3):
            p[j]=np.random.normal(loc=00.0, scale=1.0) 
        H_0=H(p,phi)
        N_md=i*10+10
        p_new,phi_new=leapfrog(p,phi)
        plt.plot(i*10+10,abs((H(p_new,phi_new)-H_0)/H_0), 'x', color='b')
        #print(p,phi)
    plt.semilogy()    
    plt.show() 




# In[42]:


def leapfrog(p,phi_l):
    # p_0_1,p_0_2,p_0_3,
    global beta 
    global f
    global x
    global J
    global N 
    global h
    global N_md
    
    eps=1/N_md
    phi_l=phi_l+eps/2*p
 
    for i in range(N_md-1):
        #print(np.sum(x**2*(f_i-phi[0]-phi[1]*x-phi[2]*x**2)/delta_f**2))
        p[0]=p[0]+eps*(beta*np.sum((f_i-phi_l[0]-phi_l[1]*x-phi_l[2]*x**2)/delta_f**2))
        p[1]=p[1]+eps*(beta*np.sum(x*((f_i-phi_l[0]-phi_l[1]*x-phi_l[2]*x**2)/delta_f**2)))
        p[2]=p[2]+eps*(beta*np.sum(x**2*(f_i-phi_l[0]-phi_l[1]*x-phi_l[2]*x**2)/delta_f**2))
        phi_l=phi_l+eps*p
        #print(p)
    p[0]=p[0]+eps*(beta*np.sum((f_i-phi_l[0]-phi_l[1]*x-phi_l[2]*x**2)/delta_f**2))
    p[1]=p[1]+eps*(beta*np.sum(x*(f_i-phi_l[0]-phi_l[1]*x-phi_l[2]*x**2)/delta_f**2))
    p[2]=p[2]+eps*(beta*np.sum(x**2*(f_i-phi_l[0]-phi_l[1]*x-phi_l[2]*x**2)/delta_f**2))
    phi_l=phi_l+eps/2*p
    #print(p)
    return p,phi_l


# input: p_0, phi_0; output: p_f,phi_f  
# code as explained on the sheet

# In[43]:


def H(p,phi_h):
    global beta 
    global J 
    global h 
    global f
    global x
    global delta_f
    #print(phi_h[0]+x*phi_h[1]+x**2*phi_h[2])
    return  np.sum(p**2)/2+beta*0.5*np.sum((f_i-(phi_h[0]+x*phi_h[1]+x**2*phi_h[2]))**2/delta_f**2)


# input p,phi: ; output: H(p,phi)

# In[44]:


def HMC(): #Does one iteration of the Markov-Chain and return phi
    global N_md
    global p_0
    global phi
    #global p
    global ar
    
    p=np.zeros(3)
    for j in range(3):
        p[j]=np.random.normal(loc=00.0, scale=1.0)
    p_l,phi_l = leapfrog(p,phi)    
    
    
    P_acc = np.exp(float(H(p,phi)-H(p_l,phi_l)))
    
        
    if P_acc > np.random.rand(): 
        
        phi = phi_l
        ar=ar+1

   


# Classical HMC-Algo, which returns the next element of the markov chain. Candidates are created with the leapfrog-algo. In our case it also keeps track of the acceptance probability with ar.

# In[45]:




def markov_chain():
    global N_cfg
    global x
    global phi
    
    
    phi_chain=[]
    for i in range(N_cfg):
        HMC()
        phi_chain.append(phi)
        #print(ar/(i+1))
    print(ar/(N_cfg))
    phi_chain=np.transpose(phi_chain)
    plt.plot(phi_chain[0], label='$\phi_0$')
    plt.plot(phi_chain[1], label='$\phi_1$')
    plt.plot(phi_chain[2], label='$\phi_2$')
    plt.legend()
    plt.show()
    phi_0=np.average(phi_chain[0])
    phi_1=np.average(phi_chain[1])
    phi_2=np.average(phi_chain[2])
    sigma_0=np.sqrt(np.average((phi_chain[0]-phi_0)**2))
    sigma_1=np.sqrt(np.average((phi_chain[1]-phi_1)**2))
    sigma_2=np.sqrt(np.average((phi_chain[2]-phi_2)**2))
    print(phi_0,phi_1,phi_2,sigma_0,sigma_1,sigma_2) 
    x=np.append(x,.134977)
    plt.plot(x,phi_0+phi_1*x+phi_2*x**2)
    plt.plot(.134977,939.5,'x', color='r')
    plt.show()

leapfrog_plot()
N_md=30
markov_chain()