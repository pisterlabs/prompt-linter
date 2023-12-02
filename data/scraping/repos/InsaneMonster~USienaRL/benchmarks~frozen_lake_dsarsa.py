#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# USienaRL is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import packages

import tensorflow
import logging
import os

# Import usienarl

from usienarl import Config, LayerType
from usienarl.utils import run_experiment, command_line_parse
from usienarl.models import DeepSARSA
from usienarl.agents import DeepSARSAAgentEpsilonGreedy, DeepSARSAAgentBoltzmann, DeepSARSAAgentDirichlet

# Import required src
# Require error handling to support both deployment and pycharm versions

try:
    from src.openai_gym_environment import OpenAIGymEnvironment
    from src.frozen_lake_refactored_environment import FrozenLakeRefactoredEnvironment
    from src.benchmark_experiment import BenchmarkExperiment
except ImportError:
    from benchmarks.src.openai_gym_environment import OpenAIGymEnvironment
    from benchmarks.src.frozen_lake_refactored_environment import FrozenLakeRefactoredEnvironment
    from benchmarks.src.benchmark_experiment import BenchmarkExperiment

# Define utility functions to run the experiment


def _define_dsarsa_model(config: Config, error_clip: bool = True) -> DeepSARSA:
    # Define attributes
    learning_rate: float = 1e-3
    discount_factor: float = 0.99
    buffer_capacity: int = 1000
    minimum_sample_probability: float = 1e-2
    random_sample_trade_off: float = 0.6
    importance_sampling_value_increment: float = 0.4
    importance_sampling_value: float = 1e-3
    # Return the _model
    return DeepSARSA("model_mse" if not error_clip else "model_huber",
                     config,
                     buffer_capacity,
                     learning_rate, discount_factor,
                     minimum_sample_probability, random_sample_trade_off,
                     importance_sampling_value, importance_sampling_value_increment,
                     error_clip)


def _define_epsilon_greedy_agent(model: DeepSARSA) -> DeepSARSAAgentEpsilonGreedy:
    # Define attributes
    summary_save_step_interval: int = 500
    weight_copy_step_interval: int = 25
    batch_size: int = 100
    exploration_rate_max: float = 1.0
    exploration_rate_min: float = 1e-3
    exploration_rate_decay: float = 1e-3
    # Return the agent
    return DeepSARSAAgentEpsilonGreedy("dsarsa_agent", model, summary_save_step_interval, weight_copy_step_interval, batch_size,
                                       exploration_rate_max, exploration_rate_min, exploration_rate_decay)


def _define_boltzmann_agent(model: DeepSARSA) -> DeepSARSAAgentBoltzmann:
    # Define attributes
    summary_save_step_interval: int = 500
    weight_copy_step_interval: int = 25
    batch_size: int = 100
    temperature_max: float = 1.0
    temperature_min: float = 1e-3
    temperature_decay: float = 1e-3
    # Return the agent
    return DeepSARSAAgentBoltzmann("dsarsa_agent", model, summary_save_step_interval, weight_copy_step_interval, batch_size,
                                   temperature_max, temperature_min, temperature_decay)


def _define_dirichlet_agent(model: DeepSARSA) -> DeepSARSAAgentDirichlet:
    # Define attributes
    summary_save_step_interval: int = 500
    weight_copy_step_interval: int = 25
    batch_size: int = 100
    alpha: float = 1.0
    dirichlet_trade_off_min: float = 0.5
    dirichlet_trade_off_max: float = 1.0
    dirichlet_trade_off_update: float = 1e-3
    # Return the agent
    return DeepSARSAAgentDirichlet("dsarsa_agent", model, summary_save_step_interval, weight_copy_step_interval, batch_size,
                                   alpha, dirichlet_trade_off_min, dirichlet_trade_off_max, dirichlet_trade_off_update)


def run(workspace: str,
        experiment_iterations: int,
        render_training: bool, render_validation: bool, render_test: bool):
    # Define the logger
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Frozen Lake environment:
    #       - general success threshold to consider the training and the experiment successful is 0.78 over 100 episodes according to OpenAI guidelines
    #       - general success threshold for refactored environment is little above (slippery) the minimum number of steps required to reach the goal
    environment_name: str = 'FrozenLake-v0'
    success_threshold: float = 0.78
    success_threshold_refactored: float = -8
    # Generate the OpenAI environment
    environment: OpenAIGymEnvironment = OpenAIGymEnvironment(environment_name)
    # Generate the refactored environment
    environment_refactored: FrozenLakeRefactoredEnvironment = FrozenLakeRefactoredEnvironment(environment_name)
    # Define Neural Network layers
    nn_config: Config = Config()
    nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()], layer_name="dense_1")
    nn_config.add_hidden_layer(LayerType.dense, [32, tensorflow.nn.relu, True, tensorflow.contrib.layers.xavier_initializer()], layer_name="dense_2")
    # Define model
    inner_model: DeepSARSA = _define_dsarsa_model(nn_config)
    # Define agents
    dsarsa_agent_epsilon_greedy: DeepSARSAAgentEpsilonGreedy = _define_epsilon_greedy_agent(inner_model)
    dsarsa_agent_boltzmann: DeepSARSAAgentBoltzmann = _define_boltzmann_agent(inner_model)
    dsarsa_agent_dirichlet: DeepSARSAAgentDirichlet = _define_dirichlet_agent(inner_model)
    # Define experiments
    experiment_epsilon_greedy: BenchmarkExperiment = BenchmarkExperiment("experiment_epsilon_greedy", success_threshold, environment,
                                                                         dsarsa_agent_epsilon_greedy)
    experiment_boltzmann: BenchmarkExperiment = BenchmarkExperiment("experiment_boltzmann", success_threshold, environment,
                                                                    dsarsa_agent_boltzmann)
    experiment_dirichlet: BenchmarkExperiment = BenchmarkExperiment("experiment_dirichlet", success_threshold, environment,
                                                                    dsarsa_agent_dirichlet)
    # Define refactored experiments
    experiment_epsilon_greedy_refactored: BenchmarkExperiment = BenchmarkExperiment("experiment_refactored_epsilon_greedy", success_threshold_refactored,
                                                                                    environment_refactored,
                                                                                    dsarsa_agent_epsilon_greedy)
    experiment_boltzmann_refactored: BenchmarkExperiment = BenchmarkExperiment("experiment_refactored_boltzmann", success_threshold_refactored,
                                                                               environment_refactored,
                                                                               dsarsa_agent_boltzmann)
    experiment_dirichlet_refactored: BenchmarkExperiment = BenchmarkExperiment("experiment_refactored_dirichlet", success_threshold_refactored,
                                                                               environment_refactored,
                                                                               dsarsa_agent_dirichlet)
    # Define experiments data
    saves_to_keep: int = 1
    plots_dpi: int = 150
    parallel: int = 10
    training_episodes: int = 100
    validation_episodes: int = 100
    training_validation_volleys: int = 20
    test_episodes: int = 100
    test_volleys: int = 10
    episode_length_max: int = 100
    # Run experiments
    run_experiment(logger=logger, experiment=experiment_epsilon_greedy,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=render_validation,
                   render_during_test=render_test,
                   iterations=experiment_iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)
    run_experiment(logger=logger, experiment=experiment_boltzmann,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=render_validation,
                   render_during_test=render_test,
                   iterations=experiment_iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)
    run_experiment(logger=logger, experiment=experiment_dirichlet,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=render_validation,
                   render_during_test=render_test,
                   iterations=experiment_iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)
    # Run refactored experiments
    run_experiment(logger=logger, experiment=experiment_epsilon_greedy_refactored,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=render_validation,
                   render_during_test=render_test,
                   iterations=experiment_iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)
    run_experiment(logger=logger, experiment=experiment_boltzmann_refactored,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=render_validation,
                   render_during_test=render_test,
                   iterations=experiment_iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)
    run_experiment(logger=logger, experiment=experiment_dirichlet_refactored,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=render_validation,
                   render_during_test=render_test,
                   iterations=experiment_iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)


if __name__ == "__main__":
    # Remove tensorflow deprecation warnings
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    # Parse the command line arguments
    workspace_path, experiment_iterations_number, cuda_devices, render_during_training, render_during_validation, render_during_test = command_line_parse()
    # Define the CUDA devices in which to run the experiment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    # Run this experiment
    run(workspace_path, experiment_iterations_number, render_during_training, render_during_validation, render_during_test)
