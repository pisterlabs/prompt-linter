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
import sys
sys.path.append('runtime-qiskit')
sys.path.append('src')
#sys.path.append('src')
import pickle
#from src.pTrace import pTraceR_num, pTraceL_num
#from src.coherence import coh_l1
#from src.kraus_maps import QuantumChannels as QCH
#from src.theoric_channels import TheoricMaps as tm

from pTrace import pTraceR_num, pTraceL_num
from coherence import coh_l1
from kraus_maps import QuantumChannels as QCH
from theoric_channels import TheoricMaps as tm


class Simulate(object):

    def __init__(self, map_name, n_qubits, list_p, epochs, step_to_start, rho_AB):
        self.list_p = list_p
        self.epochs = epochs
        self.step_to_start = step_to_start
        self.rho_AB = rho_AB
        self.coerencias_R = []
        self.map_name = map_name
        self.coerencias_L = []
        self.n_qubits = n_qubits
        self.depht = n_qubits +1
   
    def get_device(self):
        device = qml.device('qiskit.aer', wires=self.n_qubits, backend='qasm_simulator')
        return device
    
    def prepare_rho(self, theta, phi, p, gamma=None):
        if gamma == None:
            rho = self.rho_AB(theta, phi, p)
            return rho
        else:
            rho = self.rho_AB(theta, phi, p, gamma)
            return rho

    def prepare_target_op(self, theta, phi, p, gamma):
        QCH.get_target_op(self.prepare_rho(theta, phi, p))

    def plot_theoric_map(self, theta, phi):
        a = tm()
        a.plot_theoric(self.list_p,self.map_name,theta,phi)


    def general_vqacircuit_penny(self, params, n_qubits, depht=None):
        if depht == None:
            depht = self.n_qubits+1
        n = 3*self.n_qubits*(1+depht)
        #params = random_params(n)
        #params = [i for i in range(0,n)]
        #print(len(params))
        device = self.get_device()
        @qml.qnode(device, interface="torch")
        def circuit(params, M=None):
            w = [i for i in range(self.n_qubits)]
            aux = 0
            if self.n_qubits == 1:
                for j in range(depht+1):
                    qml.RX(params[aux], wires=0)
                    aux += 1
                    qml.RY(params[aux], wires=0)
                    aux += 1
                    qml.RZ(params[aux], wires=0)
                    aux += 1
                return qml.expval(qml.Hermitian(M, wires=w))
            for j in range(depht+1):
                for i in range(self.n_qubits):
                    qml.RX(params[aux], wires=i)
                    aux += 1
                    qml.RY(params[aux], wires=i)
                    aux += 1
                    qml.RZ(params[aux], wires=i)
                    aux += 1
                if j < depht:
                    for i in range(self.n_qubits-1):
                        qml.CNOT(wires=[i,i+1])
            return qml.expval(qml.Hermitian(M, wires=w))
        return circuit, params
    
    def start_things(self, depht):
        n = 3*self.n_qubits*(1+depht)
        params = np.random.normal(0,np.pi/2, n)
        params = Variable(tensor(params), requires_grad=True)
        return self.n_qubits, params, depht, n

    def cost(self, circuit, params, target_op):
        L = (1-(circuit(params, M=target_op)))**2
        return L

    def fidelidade(self, circuit, params, target_op):
        return circuit(params, M=target_op).item()

    def train(self, epocas, circuit, params, target_op, pretrain, pretrain_steps):
        opt = torch.optim.Adam([params], lr=0.1)
        best_loss = 1*self.cost(circuit, params, target_op)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_params = 1*params
        f=[]
        if pretrain:
            for start in range(pretrain_steps):
                opt.zero_grad()
                loss = self.cost(circuit, params, target_op)
                #print(epoch, loss.item())
                loss.backward()
                opt.step()
                if loss < best_loss:
                    best_loss = 1*loss
                    best_params = 1*params

        for epoch in range(epocas):
            opt.zero_grad()
            loss = self.cost(circuit, params, target_op)
            #print(epoch, loss.item())
            loss.backward()
            opt.step()
            if loss < best_loss:
                best_loss = 1*loss
                best_params = 1*params
            z = self.fidelidade(circuit, best_params, target_op)
            f.append(z)
        return best_params, f


    def general_vqacircuit_qiskit(self, n_qubits, params):
        #n = 3*self.n_qubits*(1+depht) # n=len(params)
        depht = int(len(params)/(3*self.n_qubits)-1)
        qr = QuantumRegister(self.n_qubits); qc = QuantumCircuit(qr)
        aux = 0
        for j in range(depht+1):
            for i in range(self.n_qubits):
                qc.rx(params[aux],i)
                aux += 1
                qc.ry(params[aux],i)
                aux += 1
                qc.rz(params[aux],i)
                aux += 1
            if j < depht:
                for i in range(self.n_qubits-1):
                    qc.cnot(i,i+1)
        return qc, qr

    def optmize(self, epochs, n_qubits, circuit, params, target_op, pretrain, pretrain_steps):
        best_params, f = self.train(epochs, circuit, params, target_op, pretrain, pretrain_steps)
        parametros = best_params.clone().detach().numpy()
        qc, qr = self.general_vqacircuit_qiskit(self.n_qubits, parametros)
        best_params = Variable(tensor(parametros), requires_grad=True)
        return qc, qr, best_params

    def tomograph(self):
        qstc = state_tomography_circuits(self.qc, [self.qr[0],self.qr[1]])
        nshots = 8192
        job = execute(qstc, Aer.get_backend('qasm_simulator'), shots=nshots)
        qstf = StateTomographyFitter(job.result(), qstc)
        rho = qstf.fit(method='lstsq')
        return rho

    def results(self, rho, coerencias_R, coerencias_L):
        rho_R = pTraceR_num(2,2,rho)
        rho_L = pTraceL_num(2,2,rho)
        coh_R = coh_l1(rho_R)
        coh_L = coh_l1(rho_L)
        coerencias_R.append(coh_R)
        coerencias_L.append(coh_L)

        return coerencias_L, coerencias_R

    def plots(self, list_p, coerencias_L):
        print(list_p)
        print(len(coerencias_L))
        plt.scatter(list_p,coerencias_L,label='Simulado')
        plt.xlabel(' p ')
        plt.ylabel(' Coerência ')
        #plt.legend(loc=0)
        #plt.show()

    def run_calcs(self, save, theta, phi):#, gamma=None):
        #coerencias_R = []
        coerencias_L = []
        pretrain = True
        count = 0
        #self.n_qubits = 2
        #depht = self.n_qubits + 1
        _, params, _, _ = self.start_things(self.depht)
        for p in self.list_p:
            print(f'{count} de {len(self.list_p)}')
            count += 1
            circuit, _ = self.general_vqacircuit_penny(params, self.n_qubits, self.depht)

            # defina o estado a ser preparado abaixo
            #------------------------------------------------------------
            #target_op = bpf(pi/2, 0, p)
            target_op = QCH.get_target_op(self.prepare_rho(theta, phi, p))
            #target_op = self.prepare_target_op(theta, phi, p, gamma)
            #------------------------------------------------------------

            self.qc, self.qr, params = self.optmize(self.epochs, self.n_qubits, circuit, params, target_op, pretrain, self.step_to_start)
            pretrain = False
            rho = self.tomograph()
            #print(rho)
            self.coerencias_L, self.coerencias_R = self.results(rho, self.coerencias_R, coerencias_L)
        mylist = [self.coerencias_L, self.coerencias_R, params]
        if save:
            with open(f'data/{self.map_name}/ClassTestcasa.pkl', 'wb') as f:
                pickle.dump(mylist, f)
        #plot_theoric_ad(list_p)
        
        #s = np.linspace(0,1,10)
        #z = a.theoric_rho_A_ad(np.pi/2,0,0)
        #tm.plot_theoric(self.list_p,self.theoric)
        
        #tm.plot_theoric(list_p=self.list_p,map_name=map,theta=theta,phi=phi)
        #self.prepare_plot(self.list_p)
        #print(self.map_name)
        #print(self.list_p)
        #print(theta)
        #print(phi)
        self.plot_theoric_map(theta, phi)
        self.plots(self.list_p, self.coerencias_L)
        #save = [list_p, coerencias_R, coerencias_L]
        #with open('data/BPFlist_p-coerencias_R-coerencias_L.pkl', 'wb') as f:
        #    pickle.dump(save, f)
    
    def run_sequential_bf(self, phis):
        for i in phis:
            self.run_calcs(True, pi/2, i)


def main():
    #space = np.linspace(0, 2*pi, )
    n_qubits = 3
    list_p = np.linspace(0,1,2)
    epochs = 1
    step_to_start = 1
    rho_AB = QCH.rho_AB_adg

    S = Simulate('adg', n_qubits, list_p, epochs, step_to_start, rho_AB)
    S.run_calcs(False, pi/2, 0)
    
    #phis = [0,pi,pi/1.5,pi/2,pi/3,pi/4,pi/5]
    #S.run_sequential_bf(phis)
    plt.legend(loc=1)
    plt.show()

if __name__ == "__main__":
    main()

#from src.theoric_channels import TheoricMaps as TM
#plot_theoric = TM.theoric_rho_A_bpf
#rho_AB = QCH.rho_AB_bpf
#n_qubits = 2
#list_p = np.linspace(0,1,5)
#epochs = 1
#step_to_start = 1
#
#S = Simulate('bpf/ClassTest', n_qubits, list_p, epochs, step_to_start, rho_AB, plot_theoric)
#S.run_calcs()
#print(S)