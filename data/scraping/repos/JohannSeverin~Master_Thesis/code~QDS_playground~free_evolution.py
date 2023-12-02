## Setup
import numpy as np

import sys

sys.path.append("..")

experiment_name = "control_test"
experiment_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/QDS_data/" + experiment_name


# Load devices/system
from devices.device import Resonator, Transmon
from devices.system import QubitResonatorSystem, dispersive_shift

# Load Simulation Experiment
from simulation.experiment import SchroedingerExperiment

# load Analysis tool
from analysis.auto import automatic_analysis


times = np.linspace(0, 16, 10000)

## Define devices
qubit = Transmon(
    EC=15 / 100 * 2 * np.pi, EJ=15 * 2 * np.pi, n_cutoff=15, levels=4, ng=0.0
)

resonator = Resonator(4.08677033, levels=10)    

resonator_drive = None


# Define the system
system = QubitResonatorSystem(
    qubit,
    resonator,
    coupling_strength=2 * np.pi * 0.250,
    qubit_pulse=None,
    resonator_pulse=resonator_drive,
)

# Create experiment
ground_state = system.get_states(0, 3)
excited_state = system.get_states(1, 3)

from qutip import coherent, basis, tensor

coherent_state = coherent(resonator.levels, 1)
ground_state = basis(qubit.levels, 0)

initial_state = tensor(ground_state, coherent_state)

experiment = SchroedingerExperiment(
    system,
    [initial_state],
    times,
    store_states=False,
    only_store_final=False,
    expectation_operators=[
        system.qubit_state_occupation_operator(0),
        system.qubit_state_occupation_operator(1),
        system.photon_number_operator(),
    ],
    save_path=experiment_path,
)


results = experiment.run()

analysis = automatic_analysis(results)
