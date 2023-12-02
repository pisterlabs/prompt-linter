## Setup
import numpy as np

import sys

sys.path.append("..")

experiment_name = "control_test"
experiment_path = "/mnt/c/Users/johan/OneDrive/Skrivebord/QDS_data/" + experiment_name


# Load devices/system
from devices.device import Resonator, SimpleQubit
from devices.system import QubitResonatorSystem, dispersive_shift
from devices.pulses import SquareCosinePulse

# Load Simulation Experiment
from simulation.experiment import SchroedingerExperiment

# load Analysis tool
from analysis.auto import automatic_analysis


times = np.linspace(0, 1000, 10000)

## Define devices
qubit = SimpleQubit(
    frequency=6.0,
    anharmonicity=-0.33,
)
resonator = Resonator(5.0, levels=20, kappa=2 * np.pi * 5.0 - 0.06818182 + 0.01)

resonator_drive = SquareCosinePulse(
    amplitude=0.02 * np.linspace(0, 2, 11),
    frequency=5.0 - 0.06818182 + 0.01,  # 4.9475,
    phase=0.0,
    duration=1000.0,
)


# Define the system
system = QubitResonatorSystem(
    qubit,
    resonator,
    coupling_strength=2 * np.pi * 0.250,
    resonator_pulse=resonator_drive,
)

# Create experiment
ground_state = system.get_states(0, 0)
excited_state = system.get_states(1, 0)

from qutip import coherent, basis, tensor


experiment = SchroedingerExperiment(
    system,
    [ground_state],
    times,
    store_states=False,
    only_store_final=True,
    expectation_operators=[
        system.qubit_state_occupation_operator(0),
        system.qubit_state_occupation_operator(1),
        system.photon_number_operator(),
    ],
    save_path=experiment_path,
)


results = experiment.run()

analysis = automatic_analysis(results)
