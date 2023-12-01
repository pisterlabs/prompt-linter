# # FAPS PLMAgents
import logging
import os
import random
from collections import deque

import numpy as np
import tensorflow as tf
from keras import backend as k, Input, Model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from OpenAIGym.exception import FAPSPLMEnvironmentException

logger = logging.getLogger("FAPSPLMAgents")


class FAPSTrainerException(FAPSPLMEnvironmentException):
    """
    Related to errors with the Trainer.
    """
    pass


class DQN_RC:
    """This class is the abstract class for the trainers"""

    def __init__(self, envs, brain_name, trainer_parameters, training, seed):
        """
        Responsible for collecting experiences and training a neural network model.

        :param envs: The FAPSPLMEnvironment.
        :param brain_name: The brain to train.
        :param trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param seed: Random seed.
        """

        # initialize global trainer parameters
        self.brain_name = brain_name
        self.trainer_parameters = trainer_parameters
        self.is_training = training
        self.seed = seed
        self.steps = 0
        self.last_reward = 0
        self.initialized = False

        # initialize specific DQN parameters
        self.time_slice = 4
        self.env_brains = envs
        self.state_size = 0
        self.action_size = 0
        for env_name, env in self.env_brains.items():
            self.action_size = env.action_space.n
            self.state_size = env.observation_space.n

        # self.action_space_type = envs.actionSpaceType
        self.num_layers = self.trainer_parameters['num_layers']
        self.batch_size = self.trainer_parameters['batch_size']
        self.hidden_units = self.trainer_parameters['hidden_units']
        self.replay_memory = deque(maxlen=self.trainer_parameters['memory_size'])
        self.replay_sequence = deque(maxlen=self.time_slice)
        self.gamma = self.trainer_parameters['gamma']  # discount rate
        self.epsilon = self.trainer_parameters['epsilon']  # exploration rate
        self.epsilon_min = self.trainer_parameters['epsilon_min']
        self.epsilon_decay = self.trainer_parameters['epsilon_decay']
        self.alpha = self.trainer_parameters['alpha']
        self.alpha_decay = self.trainer_parameters['alpha_decay']
        self.alpha_min = self.trainer_parameters['alpha_min']
        self.learning_rate = self.trainer_parameters['learning_rate']
        self.summary = self.trainer_parameters['summary_path']
        self.tensorBoard = tf.summary.FileWriter(logdir=self.summary)
        self.model = None
        self.target_model = None

    def __str__(self):
        return '''DQN RC Trainer'''

    @property
    def parameters(self):
        """
        Returns the trainer parameters of the trainer.
        """
        return self.trainer_parameters

    @property
    def get_max_steps(self):
        """
        Returns the maximum number of steps. Is used to know when the trainer should be stopped.
        :return: The maximum number of steps of the trainer
        """
        return self.trainer_parameters['max_steps']

    @property
    def get_step(self):
        """
        Returns the number of steps the trainer has performed
        :return: the step count of the trainer
        """
        return self.steps

    @property
    def get_last_reward(self):
        """
        Returns the last reward the trainer has had
        :return: the new last reward
        """
        return self.last_reward

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        a = Input(shape=[self.state_size * self.time_slice], name='actor_state')
        h = Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform', name="dense_actor")(a)
        h = Dropout(0.2)(h)
        for x in range(1, self.num_layers):
            h = Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform')(h)
            h = Dropout(0.2)(h)
        o = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(h)
        model = Model(inputs=a, outputs=o)
        return model

    def is_initialized(self):
        """
        check if the trainer is initialized
        """
        return self.initialized

    def _update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def initialize(self):
        """
        Initialize the trainer
        """
        self.model = self._build_model()
        self.model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate),
                           metrics=['mse'])
        print(self.model.summary())

        self.target_model = self._build_model()
        self.target_model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate),
                                  metrics=['mse'])
        print(self.target_model.summary())

        self._update_target_model()
        self.initialized = True

    def clear(self):
        """
        Clear the trainer
        """
        k.clear_session()
        self.replay_memory.clear()
        self.model = None

    def load_model_and_restore(self, model_path):
        """
        Load and restore the model from a defined path.

        :param model_path: saved model.
        """
        if os.path.exists('./' + model_path + '/DQN_RC.h5'):
            self.model = self._build_model()
            self.model.load_weights('./' + model_path + '/DQN_RC.h5')
            self.model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate),
                               metrics=['mse'])
        else:
            self.model = self._build_model()
            self.model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate),
                               metrics=['mse'])

        if os.path.exists('./' + model_path + '/DQN_RC_target.h5'):
            self.target_model = self._build_model()
            self.target_model.load_weights('./' + model_path + '/DQN_RC_target.h5')
            self.target_model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate), metrics=['mse'])
        else:
            self.target_model = self._build_model()
            self.target_model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate), metrics=['mse'])

    def increment_step(self):
        """
        Increment the step count of the trainer
        """
        self.steps = self.steps + 1

    def update_last_reward(self, rewards):
        """
        Updates the last reward
        """
        self.last_reward = rewards

    def take_action(self, observation, _env):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param observation: The BrainInfo from environment.
        :param _env: The environment.
        :return: the action array and an object as cookie
        """

        if (self.is_training and np.random.rand() <= self.epsilon) or len(self.replay_sequence) < (self.time_slice - 1):
            return np.argmax(np.random.randint(0, 2, self.action_size))
        else:
            last_elements = self.replay_sequence.copy()
            last_elements.append(observation)
            arr_last_elements = np.array(last_elements)
            tmp = arr_last_elements.reshape((1, self.state_size * self.time_slice))
            act_values = self.model.predict(tmp)

            _max = np.nanmax(act_values[0])
            indices = np.argwhere(act_values[0] == _max)
            choice = np.random.choice(indices.size)

            return indices[choice, 0]

    def add_experiences(self, observation, action, next_observation, reward, done, info):
        """
        Adds experiences to each agent's experience history.
        :param observation: the observation before executing the action
        :param action: Current executed action
        :param next_observation: the observation after executing the action
        :param reward: the reward obtained after executing the action.
        :param done: true if the episode ended.
        :param info: info after executing the action.
        """
        self.replay_sequence.append(observation)
        if len(self.replay_sequence) >= self.time_slice:
            tmp = np.array(self.replay_sequence.copy()).reshape((1, self.state_size * self.time_slice))

            next_last_elements = self.replay_sequence.copy()
            next_last_elements.append(next_observation)
            next_arr_last_elements = np.array(next_last_elements)
            next_tmp = next_arr_last_elements.reshape((1, self.state_size * self.time_slice))

            self.replay_memory.append((tmp, action, next_tmp, reward, done, info))

    def process_experiences(self, current_info, action_vector, next_info):
        """
        Checks agent histories for processing condition, and processes them as necessary.
        Processing involves calculating value and advantage targets for model updating step.
        :param current_info: Current BrainInfo.
        :param action_vector: Current executed action
        :param next_info: Next corresponding BrainInfo.
        """
        # Nothing to be done in the DQN case

    def end_episode(self):
        """
        A signal that the Episode has ended. The buffer must be reset.
        Get only called when the academy resets.
        """
        # print("End Episode...")

    def is_ready_update(self):
        """
        Returns whether or not the trainer has enough elements to run update model
        :return: A boolean corresponding to wether or not update_model() can be run
        """
        # The NN is ready to be updated if there is at least a batch in the replay memory
        # return (len(self.replay_memory) >= self.batch_size) and (len(self.replay_memory) % self.batch_size == 0)

        # The NN is ready to be updated everytime a batch is sampled
        return (self.steps > 1) and ((self.steps % self.batch_size) == 0)

    def update_model(self):
        """
        Uses the memory to update model. Run back propagation.
        """
        # TODO: update to support multiple agents. Now only one agent is supported
        num_samples = min(self.batch_size, len(self.replay_memory))
        mini_batch = random.sample(self.replay_memory, num_samples)

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for state, action, next_state, reward, done, info in mini_batch:
            state0_batch.append(state)
            state1_batch.append(next_state)
            reward_batch.append(reward)
            action_batch.append(action)
            terminal1_batch.append(0. if done else 1.)

        state0_batch = np.array(state0_batch).reshape((num_samples, self.state_size * self.time_slice))
        state1_batch = np.array(state1_batch).reshape((num_samples, self.state_size * self.time_slice))
        # terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        action_batch = np.array(action_batch)

        next_target = self.target_model.predict_on_batch(state1_batch)
        discounted_reward_batch = self.gamma * np.amax(next_target, axis=1)
        # discounted_reward_batch = discounted_reward_batch * terminal1_batch
        delta_targets = (reward_batch + discounted_reward_batch).reshape(num_samples, 1)

        q_now = self.model.predict_on_batch(state0_batch)
        q_target = q_now
        actions = np.expand_dims(action_batch, axis=1)
        np.put_along_axis(arr=q_target, indices=actions, values=delta_targets, axis=1)
        logs = self.model.train_on_batch(state0_batch, q_target)

        train_names = ['train_loss', 'train_mse']
        self._write_log(self.tensorBoard, train_names, logs, int(self.steps / self.batch_size))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.alpha > self.alpha_min:
            self.alpha *= self.alpha_decay

        # Update the target network
        if self.get_step % (4 * self.batch_size):
            self._update_target_model()

        self.write_tensorboard_value('cul_reward_mean', reward_batch.mean())

    def save_model(self, model_path):
        """
        Save the model architecture.
        :param model_path: The path where the model will be saved.
        """
        if os.path.exists(model_path):
            self.model.save(model_path + '/DQN_RC.h5')
        else:
            raise FAPSTrainerException("The model path doesn't exist. model_path : " + model_path)

        if os.path.exists(model_path):
            self.target_model.save(model_path + '/DQN_RC_target.h5')
        else:
            raise FAPSTrainerException("The model path doesn't exist. model_path : " + model_path)

    def write_summary(self):
        """
        Saves training statistics to i.e. Tensorboard.
        """
        pass

    def write_tensor_board_text(self, key, input_dict):
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param key: The name of the text.
        :param input_dict: A dictionary that will be displayed in a table on Tensorboard.
        """
        try:
            s_op = tf.summary.text(key,
                                   ([[str(x), str(input_dict[x])] for x in input_dict])
                                   )

            self.tensorBoard.add_summary(s_op, int(self.steps / self.batch_size))
        except:
            logger.info("Cannot write text summary for Tensorboard. Tensorflow version must be r1.2 or above.")

    def write_tensorboard_value(self, key, value):
        """
        Saves text to Tensorboard.
        Note: Only works on tensorflow r1.2 or above.
        :param key: The name of the text.
        :param value: A value that will bw displayed on Tensorboard.
        """
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = key
        self.tensorBoard.add_summary(summary, int(self.steps / self.batch_size))
        self.tensorBoard.flush()

    @staticmethod
    def _write_log(callback, names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.add_summary(summary, batch_no)
            callback.flush()


pass
