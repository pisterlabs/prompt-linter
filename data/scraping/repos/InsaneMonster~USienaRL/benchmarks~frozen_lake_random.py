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

import logging
import os

# Import usienarl

from usienarl.agents import RandomAgent
from usienarl.utils import command_line_parse, run_experiment

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


def _define_agent() -> RandomAgent:
    # Return the agent
    return RandomAgent("random_agent")


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
    # Define agent
    random_agent: RandomAgent = _define_agent()
    # Define experiment
    experiment_default: BenchmarkExperiment = BenchmarkExperiment("experiment_default", success_threshold, environment,
                                                                  random_agent)
    experiment_refactored: BenchmarkExperiment = BenchmarkExperiment("experiment_refactored", success_threshold_refactored, environment_refactored,
                                                                     random_agent)
    # Define experiment data (just one training volley because random agent is not trainable)
    saves_to_keep: int = 1
    plots_dpi: int = 150
    parallel: int = 10
    training_episodes: int = 100
    validation_episodes: int = 100
    training_validation_volleys: int = 1
    test_episodes: int = 100
    test_volleys: int = 10
    episode_length_max: int = 1000
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
    # Run experiment refactored
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


