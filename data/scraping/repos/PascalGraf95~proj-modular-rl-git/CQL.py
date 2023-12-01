#!/usr/bin/env python
import numpy as np
from tensorflow import keras
from keras.optimizers import Adam
from .agent_blueprint import Actor, Learner
from keras.models import load_model
from ..misc.network_constructor import construct_network
from ..misc.replay_buffer import LocalFIFOBuffer, LocalRecurrentBuffer
from ..misc.demo_reader import get_demo_files, load_demonstration, steps_from_proto
import tensorflow as tf
from keras.models import clone_model
from keras import losses
import tensorflow_probability as tfp
import os
import ray
import time
import matplotlib.pyplot as plt

tfd = tfp.distributions


@ray.remote
class CQLActor(Actor):
    def __init__(self, idx: int, port: int, mode: str,
                 interface: str,
                 preprocessing_algorithm: str,
                 preprocessing_path: str,
                 exploration_algorithm: str,
                 environment_path: str = "",
                 demonstration_path: str = "",
                 device: str = '/cpu:0'):
        super().__init__(idx, port, mode, interface, preprocessing_algorithm, preprocessing_path,
                         exploration_algorithm, environment_path, demonstration_path, device)

    def instantiate_local_buffer(self, trainer_configuration):
        if self.recurrent:
            self.local_buffer = LocalRecurrentBuffer(capacity=1000000,
                                                     agent_num=self.agent_number,
                                                     n_steps=trainer_configuration.get("NSteps"),
                                                     overlap=trainer_configuration.get("Overlap"),
                                                     sequence_length=trainer_configuration.get("SequenceLength"),
                                                     gamma=trainer_configuration.get("Gamma"))
        else:
            self.local_buffer = LocalFIFOBuffer(capacity=1000000,
                                                agent_num=self.agent_number,
                                                n_steps=trainer_configuration.get("NSteps"),
                                                gamma=trainer_configuration.get("Gamma"))
        self.network_update_frequency = trainer_configuration.get("NetworkUpdateFrequency")
        self.clone_network_update_frequency = trainer_configuration.get("SelfPlayNetworkUpdateFrequency")

    def play_one_step(self, training_step):
        if not self.samples_buffered:
            # Loop over all .demo files in the given directory.
            demonstration_paths = get_demo_files(self.demonstration_path)
            print("The following Demonstration files have been found:", demonstration_paths)
            for demo in demonstration_paths:
                print("Extracting samples from ", demo)
                behavior_spec, samples, _ = load_demonstration(demo)
                print("SAMPLE LEN:", len(samples))

                # Loop over all demo files to acquire decision and terminal steps.
                for idx, sample in enumerate(samples):
                    decision_steps, terminal_steps = steps_from_proto([sample.agent_info], behavior_spec)
                    # Append steps and actions to the local replay buffer
                    # Force the last step of a demo to be a terminal step
                    if idx == len(samples) - 1 and not len(terminal_steps.agent_id):
                        self.local_buffer.add_new_steps(decision_steps.obs, decision_steps.reward,
                                                        [a_id - self.agent_id_offset for a_id in decision_steps.agent_id],
                                                        step_type="terminal")
                        self.local_logger.track_episode(decision_steps.reward,
                                                        [a_id - self.agent_id_offset for a_id in decision_steps.agent_id],
                                                        step_type="terminal")
                    else:
                        self.local_buffer.add_new_steps(terminal_steps.obs, terminal_steps.reward,
                                                        [a_id - self.agent_id_offset for a_id in terminal_steps.agent_id],
                                                        step_type="terminal")
                        self.local_buffer.add_new_steps(decision_steps.obs, decision_steps.reward,
                                                        [a_id - self.agent_id_offset for a_id in decision_steps.agent_id],
                                                        actions=sample.action_info.continuous_actions,
                                                        step_type="decision")
                        # Track the rewards in a local logger
                        self.local_logger.track_episode(terminal_steps.reward,
                                                        [a_id - self.agent_id_offset for a_id in terminal_steps.agent_id],
                                                        step_type="terminal")

                        self.local_logger.track_episode(decision_steps.reward,
                                                        [a_id - self.agent_id_offset for a_id in decision_steps.agent_id],
                                                        step_type="decision")
                    self.local_buffer.done_agents.clear()
                print("Number of samples in the local buffer:", len(self.local_buffer))
            self.samples_buffered = True

            # region --- Demo file analysis ---
            lengths, rewards, total_episodes_played = self.local_logger.get_episode_stats()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
            fig.suptitle('Demonstration data analysis')
            ax1.hist(lengths, bins=20)
            ax1.set_title("Episode Lengths")
            ax2.scatter([np.mean(rewards)], [5], marker='o', s=20, c='r', zorder=10)
            ax2.hist(rewards, bins=20, zorder=0)
            ax2.set_title("Episode Rewards")
            plt.show()
            print("Mean Reward: ", np.mean(rewards))
            print("Total Episodes: ", total_episodes_played)
            # endregion

            return True

        # In the CQL Algorithm this method plays one whole episode each 100 training steps to evaluate the offline
        # training progress
        if training_step % 100 == 0 and self.environment_path != "NoEnv":
            while True:
                # Step acquisition (steps contain states, done_flags and rewards)
                decision_steps, terminal_steps = AgentInterface.get_steps(self.environment, self.behavior_name)
                # Preprocess steps if a respective algorithm has been activated
                decision_steps, terminal_steps = self.preprocessing_algorithm.preprocess_observations(decision_steps,
                                                                                                      terminal_steps)
                # Register terminal agents, so the hidden LSTM state is reset
                self.register_terminal_agents([a_id - self.agent_id_offset for a_id in terminal_steps.agent_id])
                # Choose the next action
                actions = self.act(decision_steps.obs,
                                   agent_ids=[a_id - self.agent_id_offset for a_id in decision_steps.agent_id],
                                   mode=self.mode)
                clone_actions = None

                # Track the rewards in a local logger
                self.local_logger.track_episode(terminal_steps.reward,
                                                [a_id - self.agent_id_offset for a_id in terminal_steps.agent_id],
                                                step_type="terminal")

                self.local_logger.track_episode(decision_steps.reward,
                                                [a_id - self.agent_id_offset for a_id in decision_steps.agent_id],
                                                step_type="decision")

                # Append steps and actions to the local replay buffer
                self.local_buffer.add_new_steps(terminal_steps.obs, terminal_steps.reward,
                                                [a_id - self.agent_id_offset for a_id in terminal_steps.agent_id],
                                                step_type="terminal")
                # If all agents are in a terminal state reset the environment
                if self.local_buffer.check_reset_condition():
                    AgentInterface.reset(self.environment)
                    self.local_buffer.done_agents.clear()
                    break
                # Otherwise, take a step in the environment according to the chosen action
                else:
                    try:
                        AgentInterface.step_action(self.environment, self.action_type,
                                                   self.behavior_name, actions, self.behavior_clone_name, clone_actions)
                    except RuntimeError:
                        print("RUNTIME ERROR")
        return True

    def act(self, states, agent_ids=None, mode="training", clone=False):
        if self.environment_path != "NoEnv":
            # Check if any agent in the environment is not in a terminal state
            active_agent_number = len(agent_ids)
            if not active_agent_number:
                return Learner.get_dummy_action(active_agent_number, self.action_shape, self.action_type)
            with tf.device(self.device):
                if self.recurrent:
                    # Set the initial LSTM states correctly according to the number of active agents
                    self.set_lstm_states(agent_ids, clone=clone)
                    # In case of a recurrent network, the state input needs an additional time dimension
                    states = [tf.expand_dims(state, axis=1) for state in states]
                    if clone:
                        (mean, log_std), hidden_state, cell_state = self.clone_actor_network(states)
                    else:
                        (mean, log_std), hidden_state, cell_state = self.actor_network(states)
                    # Update the LSTM states according to the latest network prediction
                    self.update_lstm_states(agent_ids, [hidden_state.numpy(), cell_state.numpy()], clone=clone)
                else:
                    if clone:
                        mean, log_std = self.clone_actor_network(states)
                    else:
                        mean, log_std = self.actor_network(states)
                if mode == "training" and not clone:
                    normal = tfd.Normal(mean, tf.exp(log_std))
                    actions = tf.tanh(normal.sample())
                else:
                    actions = tf.tanh(mean)
            return actions.numpy()
        return None

    def get_sample_errors(self, samples):
        """Calculates the prediction error for each state/sequence which corresponds to the initial priority in the
        prioritized experience replay buffer."""
        if self.environment_path != "NoEnv":
            if not samples:
                return None

            if self.recurrent:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                    = Learner.get_training_batch_from_recurrent_replay_batch(samples, self.observation_shapes,
                                                                             self.action_shape, self.sequence_length)
            else:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                    = Learner.get_training_batch_from_replay_batch(samples, self.observation_shapes, self.action_shape)
            if np.any(action_batch is None):
                return None

            with tf.device(self.device):
                if self.recurrent:
                    mean, log_std = self.actor_prediction_network(next_state_batch)
                else:
                    mean, log_std = self.actor_network(next_state_batch)
                normal = tfd.Normal(mean, tf.exp(log_std))
                next_actions = tf.tanh(normal.sample())
                # Critic Target Predictions
                critic_prediction = self.critic_network([*next_state_batch, next_actions])
                critic_target = critic_prediction * (1 - done_batch)

                # Train Both Critic Networks
                y = reward_batch + (self.gamma ** self.n_steps) * critic_target
                sample_errors = np.abs(y - self.critic_network([*state_batch, action_batch]))
                # In case of a recurrent agent the priority has to be averaged over each sequence according to the
                # formula in the paper
                if self.recurrent:
                    eta = 0.9
                    sample_errors = eta * np.max(sample_errors, axis=1) + (1 - eta) * np.mean(sample_errors, axis=1)
            return sample_errors
        return None

    def update_actor_network(self, network_weights):
        if self.environment_path != "NoEnv":
            if not len(network_weights):
                return
            self.actor_network.set_weights(network_weights[0])
            self.critic_network.set_weights(network_weights[1])
            if self.recurrent:
                self.actor_prediction_network.set_weights(network_weights[0])
            self.steps_taken_since_network_update = 0

    def build_network(self, network_settings, environment_parameters):
        if self.environment_path != "NoEnv":
            # Create a list of dictionaries with 4 entries, one for each network
            network_parameters = [{}, {}, {}, {}]
            # region --- Actor ---
            # - Network Name -
            network_parameters[0]['NetworkName'] = 'SAC_CQL_ActorCopy{}'.format(self.index)
            # - Network Architecture-
            network_parameters[0]['VectorNetworkArchitecture'] = network_settings["ActorVectorNetworkArchitecture"]
            network_parameters[0]['VisualNetworkArchitecture'] = network_settings["ActorVisualNetworkArchitecture"]
            network_parameters[0]['Filters'] = network_settings["Filters"]
            network_parameters[0]['Units'] = network_settings["Units"]
            # - Input / Output / Initialization -
            network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
            network_parameters[0]['Output'] = [environment_parameters.get('ActionShape'),
                                               environment_parameters.get('ActionShape')]
            network_parameters[0]['OutputActivation'] = [None, None]
            network_parameters[0]['KernelInitializer'] = "RandomUniform"
            # - Recurrent Parameters -
            network_parameters[0]['Recurrent'] = self.recurrent
            # For action calculation the recurrent actor only needs to return one output.
            network_parameters[0]['ReturnSequences'] = False
            # The network is not stateful because the batch size constantly changes. Therefor, the states need to be
            # returned and modified. The batch size thus is irrelevant.
            network_parameters[0]['Stateful'] = False
            network_parameters[0]['ReturnStates'] = True
            network_parameters[0]['BatchSize'] = None
            # endregion

            # region --- Error Prediction Actor for PER ---
            # The error prediction network is needed to calculate initial priorities for the prioritized experience
            # replay buffer.
            network_parameters[2] = network_parameters[0].copy()
            network_parameters[2]['NetworkName'] = 'SAC_CQL_ActorErrorPredictionCopy{}'.format(self.index)
            network_parameters[2]['ReturnSequences'] = True
            network_parameters[2]['ReturnStates'] = False
            network_parameters[2]['Stateful'] = False
            network_parameters[2]['BatchSize'] = None
            # endregion

            # region  --- Critic ---
            # The critic network is needed to calculate initial priorities for the prioritized experience replay
            # buffer.
            # - Network Name -
            network_parameters[1]['NetworkName'] = "SAC_CQL_CriticCopy{}".format(self.index)
            # - Network Architecture-
            network_parameters[1]['VectorNetworkArchitecture'] = network_settings["CriticVectorNetworkArchitecture"]
            network_parameters[1]['VisualNetworkArchitecture'] = network_settings["CriticVisualNetworkArchitecture"]
            network_parameters[1]['Filters'] = network_settings["Filters"]
            network_parameters[1]['Units'] = network_settings["Units"]
            network_parameters[1]['TargetNetwork'] = False
            # - Input / Output / Initialization -
            network_parameters[1]['Input'] = [*environment_parameters.get('ObservationShapes'),
                                              environment_parameters.get('ActionShape')]
            network_parameters[1]['Output'] = [1]
            network_parameters[1]['OutputActivation'] = [None]
            network_parameters[1]['KernelInitializer'] = "RandomUniform"
            # Recurrent Parameters
            network_parameters[1]['Recurrent'] = self.recurrent
            network_parameters[1]['ReturnSequences'] = True
            network_parameters[1]['Stateful'] = False
            network_parameters[1]['BatchSize'] = None
            network_parameters[1]['ReturnStates'] = False
            # endregion

            # region --- Build ---
            with tf.device(self.device):
                self.actor_network = construct_network(network_parameters[0], plot_network_model=True)
                # TODO: only construct this if PER
                self.critic_network = construct_network(network_parameters[1])
                if self.recurrent:
                    self.actor_prediction_network = construct_network(network_parameters[2])
                    # In case of recurrent neural networks, the lstm layers need to be accessible so that the hidden and
                    # cell states can be modified manually.
                    self.get_lstm_layers()
                # If there is a clone agent in the environment, instantiate another actor network for self-play.
                if self.behavior_clone_name:
                    network_parameters.append(network_parameters[0].copy())
                    network_parameters[3]['NetworkName'] = 'ActorCloneCopy{}'.format(self.index)
                    self.clone_actor_network = construct_network(network_parameters[3])
            # endregion
        return True

    def is_network_update_requested(self, training_step):
        if training_step % 100 == 99:
            return True
        return False

    def is_clone_network_update_requested(self, total_episodes):
        return False

    def connect_to_unity_environment(self):
        if self.environment_path != "NoEnv":
            super().connect_to_unity_environment()
        return True

    def set_unity_parameters(self, **kwargs):
        if self.environment_path != "NoEnv":
            self.engine_configuration_channel.set_configuration_parameters(**kwargs)

    def select_agent_interface(self, interface):
        if self.environment_path != "NoEnv":
            super().select_agent_interface(interface)

        global AgentInterface
        if interface == "MLAgentsV18":
            from ..interfaces.mlagents_v20 import MlAgentsV20Interface as AgentInterface

        elif interface == "OpenAIGym":
            from ..interfaces.openaigym import OpenAIGymInterface as AgentInterface
        else:
            raise ValueError("An interface for {} is not (yet) supported by this trainer. "
                             "You can implement an interface yourself by utilizing the interface blueprint class "
                             "in the respective folder. "
                             "After that add the respective if condition here.".format(interface))

    def read_environment_configuration(self):
        if self.environment_path != "NoEnv":
            super().read_environment_configuration()
            return True
        demonstration_paths = get_demo_files(self.demonstration_path)
        behavior_spec, samples, _ = load_demonstration(demonstration_paths[0])

        self.behavior_name, self.behavior_clone_name = AgentInterface.get_behavior_name(behavior_specs=behavior_spec)
        self.action_type = AgentInterface.get_action_type(behavior_specs=behavior_spec)
        self.action_shape = AgentInterface.get_action_shape(None, self.action_type, behavior_specs=behavior_spec)
        self.observation_shapes = AgentInterface.get_observation_shapes(behavior_specs=behavior_spec)
        self.agent_number, self.agent_id_offset = 1, 0
        self.environment_configuration = {"BehaviorName": self.behavior_name,
                                          "BehaviorCloneName": self.behavior_clone_name,
                                          "ActionShape": self.action_shape,
                                          "ActionType": self.action_type,
                                          "ObservationShapes": self.observation_shapes,
                                          "AgentNumber": self.agent_number}

    def get_exploration_configuration(self):
        return None


@ray.remote
class CQLLearner(Learner):
    ActionType = ['CONTINUOUS']
    NetworkTypes = ['Actor', 'Critic1', 'Critic2']

    def __init__(self, mode, trainer_configuration, environment_configuration, model_path=None, clone_model_path=None):
        super().__init__(trainer_configuration, environment_configuration, model_path, clone_model_path)

        # - Neural Networks -
        # The Soft Actor-Critic algorithm utilizes 5 neural networks. One actor and two critics with one target network
        # each. The actor takes in the current state and outputs an action vector which consists of a mean and standard
        # deviation for each action component. This is the only network needed for acting after training.
        # Each critic takes in the current state as well as an action and predicts its Q-Value.
        self.actor_network: keras.Model
        self.critic1: keras.Model
        self.critic_target1: keras.Model
        self.critic2: keras.Model
        self.critic_target2: keras.Model
        # A small parameter epsilon prevents math errors when log probabilities are 0.
        self.epsilon = 1.0e-6

        # CQL Parameter
        self.cql_temperature = trainer_configuration.get('CQLTemperature')
        self.cql_weight = trainer_configuration.get('CQLWeight')
        self.target_action_gap = 10
        self.cql_log_alpha = tf.Variable(tf.ones(1)*0)
        self.cql_alpha_optimizer: keras.optimizers.Optimizer

        # - Optimizer -
        self.actor_optimizer: keras.optimizers.Optimizer
        self.alpha_optimizer: keras.optimizers.Optimizer
        self.critic1_optimizer: keras.optimizers.Optimizer
        self.critic2_optimizer: keras.optimizers.Optimizer

        # - Temperature Parameter Alpha -
        # The alpha parameter similar to the epsilon in epsilon-greedy promotes exploring the environment by keeping
        # the standard deviation for each action as high as possible while still performing the task. However, in
        # contrast to epsilon, alpha is a learnable parameter and adjusts automatically.
        self.log_alpha = tf.Variable(tf.ones(1) * trainer_configuration.get('LogAlpha'),
                                     constraint=lambda x: tf.clip_by_value(x, -20, 20), trainable=True)
        self.target_entropy = -tf.reduce_sum(tf.ones(self.action_shape))

        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=trainer_configuration.get('LearningRateActor'),
                                                        clipvalue=self.clip_grad)
        self.alpha = tf.exp(self.log_alpha).numpy()

        # Construct or load the required neural networks based on the trainer configuration and environment information
        if mode == 'training':
            # Network Construction
            self.build_network(trainer_configuration.get("NetworkParameters"), environment_configuration)
            # Try to load pretrained models if provided. Otherwise, this method does nothing.
            model_key = self.get_model_key_from_dictionary(self.model_dictionary, mode="latest")
            if model_key:
                self.load_checkpoint_from_path_list(self.model_dictionary[model_key]['ModelPaths'], clone=False)
            # TODO: Implement Clone model and self-play

            # Compile Networks
            self.actor_optimizer = Adam(learning_rate=trainer_configuration.get('LearningRateActor'),
                                        clipvalue=self.clip_grad)
            self.critic1_optimizer = Adam(learning_rate=trainer_configuration.get('LearningRateCritic'),
                                          clipvalue=self.clip_grad)
            self.critic2_optimizer = Adam(learning_rate=trainer_configuration.get('LearningRateCritic'),
                                          clipvalue=self.clip_grad)
            self.cql_alpha_optimizer = Adam(learning_rate=trainer_configuration.get('LearningRateActor'),
                                            clipvalue=self.clip_grad)

        # Load trained Models
        elif mode == 'testing':
            assert model_path, "No model path entered."
            # Try to load pretrained models if provided. Otherwise, this method does nothing.
            model_key = self.get_model_key_from_dictionary(self.model_dictionary, mode="latest")
            if model_key:
                self.load_checkpoint_from_path_list(self.model_dictionary[model_key]['ModelPaths'], clone=False)

    def get_actor_network_weights(self, update_requested):
        if not update_requested:
            return []
        return [self.actor_network.get_weights(), self.critic1.get_weights()]

    def build_network(self, network_settings, environment_parameters):
        # Create a list of dictionaries with 3 entries, one for each network
        network_parameters = [{}, {}, {}]
        # region --- Actor ---
        # - Network Name -
        network_parameters[0]['NetworkName'] = "SAC_CQL_" + self.NetworkTypes[0]
        # - Network Architecture-
        network_parameters[0]['VectorNetworkArchitecture'] = network_settings["ActorVectorNetworkArchitecture"]
        network_parameters[0]['VisualNetworkArchitecture'] = network_settings["ActorVisualNetworkArchitecture"]
        network_parameters[0]['Filters'] = network_settings["Filters"]
        network_parameters[0]['Units'] = network_settings["Units"]
        # - Input / Output / Initialization -
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape'),
                                           environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = [None, None]
        network_parameters[0]['KernelInitializer'] = "RandomUniform"
        # - Recurrent Parameters -
        network_parameters[0]['Recurrent'] = self.recurrent
        # For loss calculation the recurrent actor needs to return one output per sample in the training sequence.
        network_parameters[0]['ReturnSequences'] = True
        # The actor no longer needs to be stateful due to new training process. This means the hidden and cell states
        # are reset after every prediction. Batch size thus also does not need to be predefined.
        network_parameters[0]['Stateful'] = False
        network_parameters[0]['BatchSize'] = None
        # The hidden network states are not relevant.
        network_parameters[0]['ReturnStates'] = False
        # endregion

        # region --- Critic1 ---
        # - Network Name -
        network_parameters[1]['NetworkName'] = "SAC_CQL_" + self.NetworkTypes[1]
        # - Network Architecture-
        network_parameters[1]['VectorNetworkArchitecture'] = network_settings["CriticVectorNetworkArchitecture"]
        network_parameters[1]['VisualNetworkArchitecture'] = network_settings["CriticVisualNetworkArchitecture"]
        network_parameters[1]['Filters'] = network_settings["Filters"]
        network_parameters[1]['Units'] = network_settings["Units"]
        network_parameters[1]['TargetNetwork'] = True
        # - Input / Output / Initialization -
        network_parameters[1]['Input'] = [*environment_parameters.get('ObservationShapes'),
                                          environment_parameters.get('ActionShape')]
        network_parameters[1]['Output'] = [1]
        network_parameters[1]['OutputActivation'] = [None]
        network_parameters[1]['KernelInitializer'] = "RandomUniform"
        # For image-based environments, the critic receives a mixture of images and vectors as input. If the following
        # option is true, the vector inputs will be repeated and stacked into the form of an image.
        network_parameters[1]['Vec2Img'] = False

        # - Recurrent Parameters -
        network_parameters[1]['Recurrent'] = self.recurrent
        # For loss calculation the recurrent critic needs to return one output per sample in the training sequence.
        network_parameters[1]['ReturnSequences'] = True
        # The critic no longer needs to be stateful due to new training process. This means the hidden and cell states
        # are reset after every prediction. Batch size thus also does not need to be predefined.
        network_parameters[1]['Stateful'] = False
        network_parameters[1]['BatchSize'] = None
        # The hidden network states are not relevant.
        network_parameters[1]['ReturnStates'] = False
        # endregion

        # region --- Critic2 ---
        # The second critic is an exact copy of the first one
        network_parameters[2] = network_parameters[1].copy()
        network_parameters[2]['NetworkName'] = "SAC_CQL_" + self.NetworkTypes[2]
        # endregion

        # region --- Building ---
        # Build the networks from the network parameters
        self.actor_network = construct_network(network_parameters[0], plot_network_model=True)
        self.critic1, self.critic_target1 = construct_network(network_parameters[1], plot_network_model=True)
        self.critic2, self.critic_target2 = construct_network(network_parameters[2])
        # endregion

    def forward(self, states):
        # Calculate the actors output and clip the logarithmic standard deviation values
        mean, log_std = self.actor_network(states)
        log_std = tf.clip_by_value(log_std, -20, 3)
        # Construct a normal function with mean and std and sample an action
        normal = tfd.Normal(mean, tf.exp(log_std))
        z = normal.sample()
        action = tf.tanh(z)

        # Calculate the logarithmic probability of z being sampled from the normal distribution.
        log_prob = normal.log_prob(z)
        log_prob_normalizer = tf.math.log(1 - action ** 2 + self.epsilon)
        log_prob -= log_prob_normalizer

        if self.recurrent:
            log_prob = tf.reduce_sum(log_prob, axis=2, keepdims=True)
        else:
            log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        return action, log_prob

    @ray.method(num_returns=3)
    def learn(self, replay_batch):
        if not replay_batch:
            return None, None, self.training_step

        # region --- REPLAY BATCH PREPROCESSING ---
        # In case of recurrent neural networks the batches have to be processed differently
        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes,
                                                                         self.action_shape, self.sequence_length)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = self.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes, self.action_shape)
        if np.any(action_batch is None):
            return None, None, self.training_step
        # endregion

        # region --- CRITIC TRAINING ---
        next_actions, next_log_prob = self.forward(next_state_batch)

        # Critic Target Predictions
        critic_target_prediction1 = self.critic_target1([*next_state_batch, next_actions])
        critic_target_prediction2 = self.critic_target2([*next_state_batch, next_actions])
        critic_target_prediction = tf.minimum(critic_target_prediction1, critic_target_prediction2)

        # Possible Reward DeNormalization
        if self.reward_normalization:
            critic_target_prediction = self.inverse_value_function_rescaling(critic_target_prediction)

        # Training Target Calculation with standard TD-Error + Temperature Parameter
        critic_target = (critic_target_prediction - self.alpha * next_log_prob) * (1 - done_batch)
        y = reward_batch + (self.gamma ** self.n_steps) * critic_target

        # Possible Reward Normalization
        if self.reward_normalization:
            y = self.value_function_rescaling(y)

        # region - Sample Errors -
        # Calculate Sample Errors to update priorities in Prioritized Experience Replay
        sample_errors = np.abs(y - self.critic1([*state_batch, action_batch]))
        if self.recurrent:
            eta = 0.9
            sample_errors = eta * np.max(sample_errors[:, self.burn_in:], axis=1) + \
                            (1 - eta) * np.mean(sample_errors[:, self.burn_in:], axis=1)
        # endregion

        # region --- CQL Addon ---
        # Sample 10 times the batch size of uniformly random actions
        num_repeat = 10
        random_actions = tf.random.uniform((self.batch_size * num_repeat, self.action_shape), -1, 1)

        # Add a dimension to the states batch at position one (should lead to (batch_size, 1, state_shape))
        # Then repeat this new state_batch (1, num_repeat, 1) and reshape to (batch_size * num_repeat, state_shape)
        state_batch_stack = [tf.repeat(state, num_repeat, axis=0) for state in state_batch]
        next_state_batch_stack = [tf.repeat(next_state, num_repeat, axis=0) for next_state in next_state_batch]

        # Let the actors compute actions and log probabilities for the new state and next state batches
        # Then calculate their values with both critic networks and subtract the log_pis from them.
        state_actions, state_log_prob = self.forward(state_batch_stack)
        state_stack_value1 = self.critic1([state_batch_stack, state_actions]) - state_log_prob
        state_stack_value2 = self.critic2([state_batch_stack, state_actions]) - state_log_prob

        next_state_actions, next_state_log_prob = self.forward(next_state_batch_stack)
        next_state_stack_value1 = self.critic1([next_state_batch_stack, next_state_actions]) - next_state_log_prob
        next_state_stack_value2 = self.critic2([next_state_batch_stack, next_state_actions]) - next_state_log_prob

        # Use the new state batch and the random action batch to make value predictions.
        # Then calculate simplified log probs: log(0.5**action_size)
        # Subtract the log_prob (which is a fixed value) from the random values
        random_action_values1 = tf.reshape(
            self.critic1([state_batch_stack, random_actions]) - tf.math.log(0.5 ** self.action_shape),
            (self.batch_size, num_repeat, 1))
        random_action_values2 = tf.reshape(
            self.critic2([state_batch_stack, random_actions]) - tf.math.log(0.5 ** self.action_shape),
            (self.batch_size, num_repeat, 1))

        # Reshape the current and next policy values into shape (batch_size, 10, 1)
        state_stack_value1 = tf.reshape(state_stack_value1, (self.batch_size, num_repeat, 1))
        state_stack_value2 = tf.reshape(state_stack_value2, (self.batch_size, num_repeat, 1))
        next_state_stack_value1 = tf.reshape(next_state_stack_value1, (self.batch_size, num_repeat, 1))
        next_state_stack_value2 = tf.reshape(next_state_stack_value2, (self.batch_size, num_repeat, 1))

        # Concatenate random values, current and next values along axis one (batch_size, 30, 1)
        concat_value_stack1 = tf.concat([random_action_values1, state_stack_value1, next_state_stack_value1], axis=1)
        concat_value_stack2 = tf.concat([random_action_values2, state_stack_value2, next_state_stack_value2], axis=1)

        # endregion

        # Calculate Critic 1 and 2 Loss, utilizes custom mse loss function defined in Trainer-class
        with tf.GradientTape() as tape:
            state_value1 = self.critic1([*state_batch, action_batch])
            state_value2 = self.critic2([*state_batch, action_batch])

            if self.recurrent:
                value_loss1 = tf.reduce_mean(tf.reduce_mean(tf.square(y - state_value1), axis=-1)[:, self.burn_in:])
                value_loss2 = tf.reduce_mean(tf.reduce_mean(tf.square(y - state_value2), axis=-1)[:, self.burn_in:])
            else:
                value_loss1 = tf.reduce_mean(tf.reduce_mean(tf.square(y - state_value1), axis=-1))
                value_loss2 = tf.reduce_mean(tf.reduce_mean(tf.square(y - state_value2), axis=-1))
            value_loss = (value_loss1 + value_loss2) / 2

            # First the concatenated values are scaled by the temperature parameter. Then the logsumexp is taken.
            # This corresponds to taking the exponential of the whole tensor. Then summing over axis 1.
            # This results in shape (batch_size, 1, 1). Then log is taken not changing the shape.
            # After this the mean is taken, reducing everything to a scalar
            # This again is multiplied by the temperature parameter and the cql_weight (standard: 1).
            # Then the mean value calculated by the critics for the actual state and action batches is subtracted.
            # After that weighted again.
            cql_loss1 = ((tf.reduce_mean(tf.reduce_logsumexp(concat_value_stack1 / self.cql_temperature, axis=1)) *
                         self.cql_weight * self.cql_temperature) - tf.reduce_mean(state_value1)) * self.cql_weight
            cql_loss2 = ((tf.reduce_mean(tf.reduce_logsumexp(concat_value_stack2 / self.cql_temperature, axis=1)) *
                         self.cql_weight * self.cql_temperature) - tf.reduce_mean(state_value2)) * self.cql_weight

            cql_alpha = tf.reduce_mean(tf.clip_by_value(tf.math.exp(self.cql_log_alpha), 0.0, 1000000.0))
            cql_loss1 = cql_alpha * (cql_loss1 - self.target_action_gap)
            cql_loss2 = cql_alpha * (cql_loss2 - self.target_action_gap)
            cql_alpha_loss = -(cql_loss1 + cql_loss2) * 0.5

            total_value_loss1 = value_loss1 + cql_loss1
            total_value_loss2 = value_loss2 + cql_loss2

        critic_grads = tape.gradient([total_value_loss1, total_value_loss2, cql_alpha_loss],
                                     [self.critic1.trainable_variables,
                                      self.critic2.trainable_variables,
                                      [self.cql_log_alpha]])
        self.critic1_optimizer.apply_gradients(zip(critic_grads[0], self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(critic_grads[1], self.critic2.trainable_variables))
        self.cql_alpha_optimizer.apply_gradients(zip(critic_grads[2], [self.cql_log_alpha]))
        total_value_loss = total_value_loss1 + total_value_loss2 / 2
        cql_loss = (cql_loss1 + cql_loss2) / 2
        # endregion

        # region --- ACTOR TRAINING ---
        with tf.GradientTape() as tape:
            new_actions, log_prob = self.forward(state_batch)
            critic_prediction1 = self.critic1([*state_batch, new_actions])
            critic_prediction2 = self.critic2([*state_batch, new_actions])
            critic_prediction = tf.minimum(critic_prediction1, critic_prediction2)
            policy_loss = tf.reduce_mean(self.alpha * log_prob[:, self.burn_in:] - critic_prediction[:, self.burn_in:])

        actor_grads = tape.gradient(policy_loss, self.actor_network.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_network.trainable_variables))
        # endregion

        # region --- TEMPERATURE PARAMETER TRAINING ---
        """
        with tf.GradientTape() as tape:
            alpha_loss = - tf.reduce_mean(self.log_alpha * (log_prob + self.target_entropy))
        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        """
        self.alpha = tf.exp(self.log_alpha).numpy()
        # endregion

        self.training_step += 1
        self.steps_since_actor_update += 1
        self.sync_models()

        return {'Losses/Loss': policy_loss + total_value_loss, 'Losses/PolicyLoss': policy_loss,
                'Losses/ValueLoss': value_loss, 'Losses/CQLLoss': cql_loss, 'Losses/CQLAlpha': cql_alpha.numpy(),
                'Losses/Alpha': tf.reduce_mean(self.alpha).numpy()}, sample_errors, self.training_step

    def sync_models(self):
        if self.sync_mode == "hard_sync":
            if not self.training_step % self.sync_steps and self.training_step > 0:
                self.critic_target1.set_weights(self.critic1.get_weights())
                self.critic_target2.set_weights(self.critic2.get_weights())
        elif self.sync_mode == "soft_sync":
            self.critic_target1.set_weights([self.tau * weights + (1.0 - self.tau) * target_weights
                                             for weights, target_weights in zip(self.critic1.get_weights(),
                                                                                self.critic_target1.get_weights())])
            self.critic_target2.set_weights([self.tau * weights + (1.0 - self.tau) * target_weights
                                             for weights, target_weights in zip(self.critic2.get_weights(),
                                                                                self.critic_target2.get_weights())])
        else:
            raise ValueError("Sync mode unknown.")

    def load_checkpoint_from_path_list(self, model_paths, clone=False):
        if not clone:
            for file_path in model_paths:
                if "Critic1" in file_path:
                    self.critic1 = load_model(file_path, compile=False)
                    self.critic_target1 = clone_model(self.critic1)
                    self.critic_target1.set_weights(self.critic1.get_weights())
                elif "Critic2" in file_path:
                    self.critic2 = load_model(file_path, compile=False)
                    self.critic_target2 = clone_model(self.critic2)
                    self.critic_target2.set_weights(self.critic2.get_weights())
                elif "Actor" in file_path:
                    self.actor_network = load_model(file_path, compile=False)
            if not self.actor_network:
                raise FileNotFoundError("Could not find all necessary model files.")
            if not self.critic1 or not self.critic2:
                print("WARNING: Critic models for CQL not found. "
                      "This is not an issue if you're planning to only test the model.")

    def save_checkpoint(self, path, running_average_reward, training_step, save_all_models=False,
                        checkpoint_condition=True):
        if not checkpoint_condition:
            return
        self.actor_network.save(
            os.path.join(path, "SAC_CQL_Actor_Step{:06d}_Reward{:.2f}.h5".format(training_step,
                                                                                 running_average_reward)))
        self.critic1.save(
            os.path.join(path, "SAC_CQL_Critic1_Step{:06d}_Reward{:.2f}.h5".format(training_step,
                                                                                   running_average_reward)))
        self.critic2.save(
            os.path.join(path, "SAC_CQL_Critic2_Step{:06d}_Reward{:.2f}.h5".format(training_step,
                                                                                   running_average_reward)))

    def boost_exploration(self):
        self.log_alpha = tf.Variable(tf.ones(1) * -0.7,
                                     constraint=lambda x: tf.clip_by_value(x, -10, 20), trainable=True)
        return True
