# # FAPS PLMAgents

import logging
import os
import random
from collections import deque

import keras.backend as k
import numpy as np
import tensorflow as tf
from keras import Input
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam, RMSprop

from OpenAIGym.exception import FAPSPLMEnvironmentException

logger = logging.getLogger("FAPSPLMAgents")


class FAPSTrainerException(FAPSPLMEnvironmentException):
    """
    Related to errors with the Trainer.
    """
    pass


class A2C(object):
    """This class is the abstract class for the faps trainers"""

    def __init__(self, envs, brain_name, trainer_parameters, training, seed):
        """
        Responsible for collecting experiences and training a neural network model.

        :param envs: The FAPSPLMEnvironment.
        :param brain_name: The brain to train.
        :param trainer_parameters: The parameters for the trainer (dictionary).
        :param training: Whether the trainer is set for training.
        :param seed: Random seed.
        """
        self.brain_name = brain_name
        self.trainer_parameters = trainer_parameters
        self.is_training = training
        self.seed = seed
        self.steps = 0
        self.last_reward = 0
        self.initialized = False

        # initialize specific DDPG parameters
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
        self.learning_rate = self.trainer_parameters['learning_rate']
        self.summary = self.trainer_parameters['summary_path']
        self.tensorBoard = tf.summary.FileWriter(logdir=self.summary)
        self.actor_model = None
        self.critic_model = None

    def __str__(self):
        return '''Deep Deterministic Policy Gradient Trainer'''

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
        return self.trainer_parameters['max_steps'] * self.trainer_parameters['num_epoch']

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

    def get_actor_optimizer(self):
        """
        Actor Optimizer
        """
        action_gdts = k.placeholder(shape=(None, self.action_size))
        params_grad = tf.gradients(self.actor_model.output, self.actor_model.trainable_weights, -action_gdts)
        grads = zip(params_grad, self.actor_model.trainable_weights)
        return k.function([self.actor_model.input, action_gdts],
                          [tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)][1:])

    def _create_actor_model(self):
        a = Input(shape=[self.state_size * self.time_slice])
        h = Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform')(a)
        h = Dropout(0.2)(h)
        for x in range(1, self.num_layers):
            h = Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform')(h)
            h = Dropout(0.2)(h)
        o = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(h)
        model = Model(inputs=a, outputs=o)
        return model

    def _create_critic_model(self):
        a = Input(shape=[self.state_size * self.time_slice])
        h = Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform')(a)
        h = Dropout(0.2)(h)
        for x in range(1, self.num_layers):
            h = Dense(self.hidden_units, activation='relu', kernel_initializer='he_uniform')(h)
            h = Dropout(0.2)(h)
        o = Dense(1, activation='linear', kernel_initializer='he_uniform')(h)
        model = Model(inputs=a, outputs=o)
        return model

    def is_initialized(self):
        """
        check if the trainer is initialized
        """
        return self.initialized

    def initialize(self):
        """
        Initialize the trainer
        """
        self.actor_model = self._create_actor_model()
        self.actor_model.compile(loss='categorical_crossentropy', metrics=['categorical_crossentropy'],
                                 optimizer=Adam(lr=self.learning_rate))
        print('\n##### Actor Model ')
        print(self.actor_model.summary())

        self.critic_model = self._create_critic_model()
        self.critic_model.compile(loss='mse', metrics=['mse'], optimizer=RMSprop(lr=self.learning_rate))
        print('\n##### Critic Model ')
        print(self.critic_model.summary())

        self.initialized = True

    def clear(self):
        """
        Clear the trainer
        """
        k.clear_session()
        self.replay_memory.clear()
        self.actor_model = None
        self.critic_model = None

    def load_model_and_restore(self, model_path):
        """
        Load and restore the model from a defined path.

        :param model_path: Random seed.
        """
        self.actor_model = self._create_actor_model()
        if os.path.exists('./' + model_path + '/A2C_actor_model.h5'):
            self.actor_model.load_weights('./' + model_path + '/A2C_actor_model.h5')
        self.actor_model.compile(loss='categorical_crossentropy', metrics=['categorical_crossentropy'],
                                 optimizer=Adam(lr=self.learning_rate))

        self.critic_model = self._create_critic_model()
        if os.path.exists('./' + model_path + '/A2C_critic_model.h5'):
            self.critic_model.load_weights('./' + model_path + '/A2C_critic_model.h5')
        self.critic_model.compile(loss='mse', metrics=['mse'], optimizer=RMSprop(lr=self.learning_rate))

    def increment_step(self):
        """
        Increment the step count of the trainer
        """
        self.steps = self.steps + 1

    def update_last_reward(self, reward):
        """
        Updates the last reward
        """
        self.last_reward = reward

    def take_action(self, observation, _env):
        """
        Decides actions given state/observation information, and takes them in environment.
        :param observation: The BrainInfo from environment.
        :param _env: The environment.
        :return: the action array and an object as cookie
        """
        if len(self.replay_sequence) < self.time_slice - 1:
            return np.argmax(np.random.randint(0, 2, self.action_size))
        else:
            last_elements = self.replay_sequence.copy()
            last_elements.append(observation)
            arr_last_elements = np.array(last_elements)
            tmp = arr_last_elements.reshape((1, self.state_size * self.time_slice))
            policy = self.actor_model.predict(tmp).ravel()
            return np.random.choice(np.arange(self.action_size), 1, p=policy)[0]

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
        return (self.steps > 1) and ((self.steps % self.batch_size) == 0)

    def update_model(self):
        """
        Uses the memory to update model. Run back propagation.
        """
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
        terminal1_batch = np.array(terminal1_batch).reshape((num_samples, 1))
        reward_batch = np.array(reward_batch).reshape((num_samples, 1))
        action_batch = np.array(action_batch).reshape((num_samples, 1))

        # Compute values
        values = self.critic_model.predict_on_batch([state0_batch])
        next_values = self.critic_model.predict_on_batch([state1_batch])
        self.write_tensorboard_value('values_mean', values.mean())

        # Compute advantages and critic target
        discounted_reward_batch = self.gamma * next_values
        discounted_reward_batch = discounted_reward_batch * terminal1_batch
        advantages = (reward_batch + discounted_reward_batch - values).reshape(num_samples, 1)
        critic_targets = (reward_batch + discounted_reward_batch).reshape(num_samples, 1)

        # Train the critic
        critic_logs = self.critic_model.train_on_batch([state0_batch], critic_targets)
        train_names = ['critic_train_loss', 'critic_train_mse']
        self._write_log(self.tensorBoard, train_names, critic_logs, int(self.steps / self.batch_size))

        # Train the actor
        train_action_batch = np.zeros((num_samples, self.action_size))
        np.put_along_axis(arr=train_action_batch, indices=action_batch, values=advantages, axis=1)
        actor_logs = self.critic_model.train_on_batch([state0_batch], advantages)
        train_names = ['actor_train_loss', 'actor_train_categorical_crossentropy']
        self._write_log(self.tensorBoard, train_names, actor_logs, int(self.steps / self.batch_size))

    def save_model(self, model_path):
        """
        Save the model architecture to i.e. Tensorboard.
        :param model_path: The path where the model will be saved.
        """
        if os.path.exists('./' + model_path):
            self.actor_model.save('./' + model_path + '/A2C_actor_model.h5')
            self.critic_model.save('./' + model_path + '/A2C_critic_model.h5')
        else:
            raise FAPSTrainerException("The model path doesn't exist. model_path : " + './' + model_path)

    def write_summary(self):
        """
        Saves training statistics to i.e. Tensorboard.
        """
        # print(self.model.summary())
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
        except AttributeError:
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
