from torch.autograd import Variable
import pennylane as qml
from qiskit import *
from qiskit import Aer, execute
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit import QuantumRegister
from qiskit import QuantumCircuit
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import tensor
from numpy import pi
import os
from matplotlib.widgets import Slider, Button
import sys
sys.path.append('runtime-qiskit')
sys.path.append('src')
#sys.path.append('src')
import pickle
import ipywidgets as widgets
from IPython.display import display
#from src.pTrace import pTraceR_num, pTraceL_num
#from src.coherence import coh_l1
#from src.kraus_maps import QuantumChannels as QCH
#from src.theoric_channels import TheoricMaps as tm

from pTrace import pTraceR_num, pTraceL_num
from coherence import coh_l1
from kraus_maps import QuantumChannels as QCH
from kraus_maps import get_list_p_noMarkov
from theoric_channels import TheoricMaps as tm
from numpy import cos, sin, sqrt, pi, exp
from sympy import *
import numpy as np

init_printing(use_unicode=True)
from matplotlib import pyplot as plt
#%matplotlib inline
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum import TensorProduct
import scipy.interpolate
import platform

def cb(d, j):
    cbs = zeros(d,1); cbs[j] = 1
    return cbs
def proj(psi):
    return psi*Dagger(psi)

def coh_l1(rho):
    d = rho.shape[0]; C = 0
    for j in range(0,d-1):
        for k in range(j+1,d):
            C += abs(rho[j,k])
    return 2*C

def Pauli(j):
    if j == 0:
        return Matrix([[1,0],[0,1]])
    elif j == 1:
        return Matrix([[0,1],[1,0]])
    elif j == 2:
        return Matrix([[0,-1j],[1j,0]])
    elif j == 3:
        return Matrix([[1,0],[0,-1]])                           

class Calculate(object):

    def __init__(self):
        pass

   
    def phase_flip(J):
        def K_0(J):
            return sqrt(1-J/2)*Pauli(0)
        def K_1(J):
            return sqrt(J/2)*Pauli(3)
        print(K_0, K_1)
        return K_0, K_1
    
    def RHO_t_Ana(state,J):
        tp1 = TP(K_0(J),K_1(J))
        tp2 = TP(K_1(J),K_0(J))
        return tp1*proj(state)*tp1.T + tp2*proj(state)*tp2.T

    def RHO_t_Ana(state,J):
        tp1 = TP(K_0(J),K_1(J))
        tp2 = TP(K_1(J),K_0(J))
        return tp1*proj(state)*tp1.T + tp2*proj(state)*tp2.T

# def TP(a,b):
    # return TensorProduct(a,b)
# def RHO_t_NM(state,J):
    # tp1 = TP(K_0(J),K_1(J))
    # tp2 = TP(K_1(J),K_0(J))
    # return tp1*proj(state)*tp1.T + tp2*proj(state)*tp2.T