import os
import argparse
import functools
import pickle
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
import qiskit.providers.aer.noise as noise

import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.plugins import tq2qiskit
from torchquantum import switch_little_big_endian_state

from qutip.states import coherent
from qutip.visualization import plot_wigner
from qutip.wigner import wigner


class Op1ToWires(tq.QuantumModule):

    def __init__(self, op, wires: list, has_params=False, trainable=False):
        super().__init__()
        self.wires = wires
        self.op = op
        self.ops_all = tq.QuantumModuleList()
        for k in range(len(wires)):
            self.ops_all.append(op(has_params=has_params, trainable=trainable))

    @tq.static_support
    def forward(self, q_device):
        for i, k in enumerate(self.wires):
            self.ops_all[i](q_device, wires=k)


class QuantumAutoencoder(tq.QuantumModule):

    def __init__(self, dim_input, dim_latent, num_blocks):
        super().__init__()
        self.dim_input = dim_input
        self.dim_latent = dim_latent
        self.dim_trash = dim_input - dim_latent
        self.num_blocks = num_blocks
        self.num_qubits = 1 + self.dim_input + self.dim_trash
        self.ry_layers = tq.QuantumModuleList()
        for _ in range(self.num_blocks + 1):
            self.ry_layers.append(
                Op1ToWires(
                    op=tq.U3,
                    wires=[i for i in range(self.dim_input)],
                    has_params=True,
                    trainable=True,
                )
            )
        
    @tq.static_support
    def forward(self, q_device):
        # encoder
        self.ry_layers[0](q_device)
        for layer in range(1, self.num_blocks + 1):
            for qubit in reversed(range(self.dim_input - 1)):
                tqf.cnot(q_device, wires=[qubit, qubit + 1], static=self.static_mode, parent_graph=self.graph)
            self.ry_layers[layer](q_device)

        if self.training:
            tqf.hadamard(q_device, wires=self.num_qubits - 1, static=self.static_mode, parent_graph=self.graph)
            for i in range(self.dim_latent, self.dim_latent + self.dim_trash):
                tqf.cswap(q_device, wires=[self.num_qubits - 1, i, i + self.dim_trash], static=self.static_mode, parent_graph=self.graph)
            tqf.hadamard(q_device, wires=self.num_qubits - 1, static=self.static_mode, parent_graph=self.graph)
            return measure_wire_computational(q_device, self.num_qubits - 1)[:, 1]


class SwapTest(tq.QuantumModule):

    def __init__(self, n_wires_state):
        super().__init__()
        self.n_wires = n_wires_state
        self.num_qubits = 2 * n_wires_state + 1
    
    @tq.static_support
    def forward(self, q_device):
        tqf.hadamard(q_device, wires=self.num_qubits - 1, static=self.static_mode, parent_graph=self.graph)
        for i in range(self.n_wires):
            tqf.cswap(q_device, wires=[self.num_qubits - 1, i, i + self.n_wires], static=self.static_mode, parent_graph=self.graph)
        tqf.hadamard(q_device, wires=self.num_qubits - 1, static=self.static_mode, parent_graph=self.graph)


class StateGenerator:
    """Dataloader for training the autoencoder."""

    def __init__(self, state_type, num_qubits, shuffle=False, batch_size=1):
        self.state_dim = 2 ** num_qubits
        if state_type == 'cat':
            dataset = self.get_cat_states()
        self.idx = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def get_cat_states(self, num_data=32):
        dataset = []
        N = self.state_dim
        for alpha in np.linspace(3, 4, num_data):
            N = self.state_dim
            psi = (coherent(N, -alpha) + coherent(N, alpha))
            psi /= psi.norm()
            dataset.append(psi.full().squeeze())
        return np.array(dataset)
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.dataset):
            self.idx = 0
            if self.shuffle:
                np.random.shuffle(self.dataset)
            raise StopIteration

        start = self.idx
        self.idx += self.batch_size
        return torch.tensor(self.dataset[start: self.idx], dtype=torch.complex64)


def measure_wire_computational(qdev, wire):
    """Measure a wire in the computational basis."""
    all_dims = np.arange(qdev.states.dim())
    states = qdev.states
    # compute magnitude
    state_mag = torch.abs(states) ** 2
    # compute marginal magnitude
    reduction_dims = np.delete(all_dims, [0, wire + 1])
    probs = state_mag.sum(list(reduction_dims))

    return probs


def ket0_states(num_qubits, device):
    """Return the |00...0> state for the given number of qubits."""
    ket0 = torch.zeros(2 ** num_qubits, dtype=torch.complex64, device=device)
    ket0[0] = 1. + 0.j
    return ket0


def test_swap():
    n_wires = 3
    q_device = tq.QuantumDevice(n_wires=7)
    # q_device.reset_states(bsz=1)
    state1 = torch.tensor([0, 0, 0, 0, 0, 1, 0, 1], dtype=torch.complex64) / np.sqrt(2)
    state2 = torch.tensor([0, 0, 0, 0, 0, 1, 0, 1], dtype=torch.complex64) / np.sqrt(2)
    state3 = torch.tensor([1, 0], dtype=torch.complex64)
    state4 = torch.kron(state1, state2)
    input_state = torch.kron(state4, state3)
    q_device.set_states(input_state[None])
    tq_model = SwapTest(n_wires)
    tq_model.train()
    tq_model(q_device)
    circ = tq2qiskit(q_device, tq_model)
    print(circ)
    states = q_device.states
    state_mag = torch.abs(states) ** 2
    all_dims = np.arange(states.dim())
    reduction_dims = np.delete(all_dims, [0, 7])
    probs = state_mag.sum(list(reduction_dims))
    print(probs)


def plot_wigner_psi_phi(psi, alpha_max=7.5):
    fig = plt.figure(figsize=(9,9))

    widths = [6,3]
    heights = [6,3]
    spec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                              height_ratios=heights)

    x = np.linspace(-alpha_max,alpha_max,200)
    wig = wigner(psi,x,x)
    psi_x = np.sum(wig,axis=0)
    psi_p = np.sum(wig,axis=1)


    ax = fig.add_subplot(spec[0,0])
    plot_wigner(psi,fig=fig,ax=ax,alpha_max = alpha_max)
    ax = fig.add_subplot(spec[0,1])
    base = plt.gca().transData
    rot = transforms.Affine2D().rotate_deg(90)
    ax.plot(x,-psi_p, transform = rot+base)
    ax.set_xticks([])
    ax.set_ylim(-alpha_max,alpha_max)
    
    ax = fig.add_subplot(spec[1,0])
    ax.plot(x,psi_x)
    ax.set_yticks([])
    ax.set_xlim(-alpha_max,alpha_max)

    
def train(args):
    # prepare
    tq_model = QuantumAutoencoder(args.num_input, args.num_hidden, 2).to(args.device)
    tq_model.train()
    q_device = tq.QuantumDevice(n_wires=2 * args.num_input - args.num_hidden + 1, device=args.device)
    q_device.reset_states(bsz=1)
    data_loader = StateGenerator('cat', args.num_input, shuffle=True, batch_size=1)
    optimizer = optim.Adam(tq_model.parameters(), lr=args.lr, weight_decay=0)

    # train
    for epoch in range(args.num_epochs):
        for i, data in enumerate(data_loader):
            data = data.squeeze().to(args.device)
            data = torch.kron(data, ket0_states(args.num_input - args.num_hidden + 1, args.device))[None]
            q_device.set_states(data)
            optimizer.zero_grad()
            output = tq_model(q_device)
            output.backward()
            optimizer.step()
            # print('loss:', output.item())

    # evaluate
    q_device = tq.QuantumDevice(n_wires=args.num_input)
    tq_model.cpu().eval()
    circuit = tq2qiskit(q_device, tq_model)
    input_states = data_loader.get_cat_states().squeeze()
    fidelities = []
    for input_state in input_states:
        test_circuit = QuantumCircuit(args.num_input)
        input_state = switch_little_big_endian_state(input_state)
        test_circuit.initialize(input_state, range(args.num_input))
        test_circuit = test_circuit.compose(circuit)
        test_circuit.barrier()
        for i in range(args.num_hidden, args.num_input):
            test_circuit.reset(i)
        test_circuit.barrier()
        test_circuit = test_circuit.compose(circuit.inverse())
        backend = AerSimulator()
        t_circuit = transpile(test_circuit, backend, optimization_level=0)
        result = Statevector(t_circuit).data
        fidelity = np.abs(np.dot(result.conj(), input_state)) ** 2
        fidelities.append(fidelity)

    fidelity = np.mean(fidelities)

    return fidelity, circuit

    # test_circuit.measure_all()
    # state_1 = Statevector(test_circuit).data
    # print()
    # noise_model = noise.NoiseModel()
    # error_1 = noise.depolarizing_error(0.1, 1)
    # noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'i', 'x', 'y', 'z', 'h', 's', 't', 'sdg', 'tdg'])
    # backend = AerSimulator()
    # t_circuit = transpile(test_circuit, backend, optimization_level=0)
    # # results = backend.run(t_circuit, shots=1024).result().get_counts()
    # state_2 = Statevector(t_circuit).data
    # print(np.abs(np.dot(state_1.conj(), state_2)) ** 2)

    # q_device = tq.QuantumDevice(n_wires=2 * args.num_input - args.num_hidden + 1)
    # tq_model.cpu()
    # circuit = tq2qiskit(q_device, tq_model)
    # test_circuit = QuantumCircuit(2 * args.num_input - args.num_hidden + 1, 1)
    # input_state = data_loader.get_cat_states().squeeze()
    # input_state = switch_little_big_endian_state(input_state)
    # test_circuit.initialize(input_state, range(args.num_input))
    # test_circuit = test_circuit.compose(circuit)
    # test_circuit.barrier()
    # test_circuit.measure(2 * args.num_input - args.num_hidden, 0)
    # backend = AerSimulator()
    # counts = backend.run(transpile(test_circuit, backend), shots=1024).result().get_counts()
    # print(counts)
    

def generate_train_circuits(args):
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    total_circuits = 0
    while total_circuits < args.circuit_num:
        fidelity, test_circuit = train(args)
        if fidelity > 0.9:
            print(fidelity)
            total_circuits += 1
            save_path = os.path.join(args.out_path, f'ae_{total_circuits}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(test_circuit, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_input', type=int, default=6, help='Dimension of input state.')
    parser.add_argument('--num_hidden', type=int, default=4, help='Dimension of encoded state.')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs to train.')
    parser.add_argument('--out_path', type=str, default='../environments/circuits/autoencoder_6l', help='Output circuit dir.')
    parser.add_argument('--circuit_num', type=int, default=1, help='Number of circuits for mitigation.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generate_train_circuits(args)