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
from usienarl.models import ProximalPolicyOptimization
from usienarl.agents.ppo_agent import PPOAgent

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


def _define_ppo_model(actor_config: Config, critic_config: Config) -> ProximalPolicyOptimization:
    # Define attributes
    learning_rate_policy: float = 3e-4
    learning_rate_advantage: float = 1e-3
    discount_factor: float = 0.99
    value_steps_per_update: int = 80
    policy_steps_per_update: int = 80
    minibatch_size: int = 32
    lambda_parameter: float = 0.97
    clip_ratio: float = 0.2
    target_kl_divergence: float = 1e-2
    # Return the model
    return ProximalPolicyOptimization("model", actor_config, critic_config, discount_factor,
                                      learning_rate_policy, learning_rate_advantage,
                                      value_steps_per_update, policy_steps_per_update,
                                      minibatch_size,
                                      lambda_parameter,
                                      clip_ratio,
                                      target_kl_divergence)


def _define_agent(model: ProximalPolicyOptimization) -> PPOAgent:
    # Define attributes
    update_every_episodes: int = 1000
    # Return the agent
    return PPOAgent("ppo_agent", model, update_every_episodes)


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
    inner_model: ProximalPolicyOptimization = _define_ppo_model(actor_config=nn_config, critic_config=nn_config)
    # Define agent
    ppo_agent: PPOAgent = _define_agent(inner_model)
    # Define experiment
    experiment_default: BenchmarkExperiment = BenchmarkExperiment("experiment_default", success_threshold, environment,
                                                                  ppo_agent)
    # Define refactored experiment
    experiment_refactored: BenchmarkExperiment = BenchmarkExperiment("experiment_refactored", success_threshold_refactored,
                                                                     environment_refactored,
                                                                     ppo_agent)
    # Define experiments data
    saves_to_keep: int = 1
    plots_dpi: int = 150
    parallel: int = 10
    training_episodes: int = 5000
    validation_episodes: int = 100
    training_validation_volleys: int = 20
    test_episodes: int = 100
    test_volleys: int = 10
    episode_length_max: int = 100
    # Run experiment
    run_experiment(logger=logger, experiment=experiment_default,
                   file_name=__file__, workspace_path=workspace,
                   training_volleys_episodes=training_episodes, validation_volleys_episodes=validation_episodes,
                   training_validation_volleys=training_validation_volleys,
                   test_volleys_episodes=test_episodes, test_volleys=test_volleys,
                   episode_length=episode_length_max, parallel=parallel,
                   render_during_training=render_training, render_during_validation=render_validation,
                   render_during_test=render_test,
                   iterations=experiment_iterations, saves_to_keep=saves_to_keep, plots_dpi=plots_dpi)
    # Run refactored experiment
    run_experiment(logger=logger, experiment=experiment_refactored,
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
