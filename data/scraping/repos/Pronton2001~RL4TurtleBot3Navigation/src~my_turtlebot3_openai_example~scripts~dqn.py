from collections import namedtuple, deque
from itertools import count
import random
from turtle import update
import numpy as np
import gym
import math
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation
from tensorflow import losses
from keras.models import model_from_yaml
import rospy
import os
import json
import time
from std_msgs.msg import Float32MultiArray

EPS_START=1
EPS_END=0.05
# EPS_DECAY= 200
EPS_DECAY= .99
# TARGET_UPDATE=10
START_USE_TARGET_STEPS = 2000
def get_huber_loss_fn(**huber_loss_kwargs):

    def custom_huber_loss(y_true, y_pred):
        return losses.huber_loss(y_true, y_pred, **huber_loss_kwargs)

    return custom_huber_loss
from openai_ros.task_envs.turtlebot3_nav import turtlebot3_nav

def createQNet(input_dim, n_actions, alpha, alpha_decay):
    # model = Sequential()
    # model.add(Dense(24, input_dim=input_dim, activation='tanh'))
    # model.add(Dense(48, activation='tanh'))
    # model.add(Dense(n_actions, activation='linear'))
    # model.compile(loss=get_huber_loss_fn(delta=0.1), optimizer=Adam(lr=alpha, decay=alpha_decay))
    model = Sequential()
    dropout = 0.2

    model.add(Dense(64, input_shape=(input_dim,), activation='relu', kernel_initializer='lecun_uniform'))

    model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
    model.add(Dropout(dropout))

    model.add(Dense(n_actions, kernel_initializer='lecun_uniform'))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer=RMSprop(lr=alpha, rho=0.9, epsilon=1e-06))
    model.summary() 
    return model

class DQNSolver():
    def __init__(self, env, n_observations, n_actions, max_env_steps=None, gamma=.99, alpha=0.01, alpha_decay=0.01, batch_size=64, update_target_steps = 10, load_trained_model = False, load_episode = 0):

        self.epsilon = EPS_START
        self.load_trained_model = load_trained_model
        self.load_episode = load_episode
        self.env = env
        self.memory = deque(maxlen=100000)
        self.n_obs = n_observations
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps
        self.update_target_steps = update_target_steps
        self.policy_net = createQNet(self.n_obs, self.n_actions, self.alpha, self.alpha_decay)
        self.target_net = createQNet(self.n_obs, self.n_actions, self.alpha, self.alpha_decay)
        self.dirPath = '/home/pronton/catkin_ws/src/openai_examples_projects/my_turtlebot3_openai_example/scripts/dqn2_models/'
        self.pub_info = rospy.Publisher('info', Float32MultiArray, queue_size=5)
        # self.pub_state = rospy.Publisher('state', Float32MultiArray, queue_size=5)
        if self.load_trained_model:
            model_name = "dqn" + str(self.load_episode)
            self.load(model_name, models_dir_path=self.dirPath)
            # self.policy_net.set_weights(load_model(h+ ).get_weights())
            with open(self.dirPath + "dqn" + str(self.load_episode)+'_eps.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')
    
    def getQvalue(self, reward, next_target, done):
        if done:
            return reward
        else:
            return reward + self.gamma * np.amax(next_target)

    def updateTargetModel(self):
        self.target_net.set_weights(self.policy_net.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, eps_threshold):
        if np.random.random() > eps_threshold: # best action policy
            q_value = self.policy_net.predict(np.reshape(state, [1, self.n_obs]))
            self.q_value = q_value
            return np.argmax(q_value[0])
        else: # random action
            self.q_value = np.zeros(self.n_actions)
            return random.randrange(self.n_actions)
            # return self.env.action_space.sample()

    # def preprocess_state(self, state):
    #     return np.reshape(state, [1, self.input_dim])
    def trainModel(self, target=False):
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.n_obs), dtype=np.float)
        Y_batch = np.empty((0, self.n_actions), dtype=np.float)

        for i in range(self.batch_size):
            states = mini_batch[i][0]
            actions = mini_batch[i][1]
            rewards = mini_batch[i][2]
            next_states = mini_batch[i][3]
            dones = mini_batch[i][4]

            # q_value = self.policy_net.predict(states.reshape(1, len(states)))
            q_value = self.policy_net.predict(np.reshape(states, [1, self.n_obs]))
            self.q_value = q_value

            if target:
                next_target = self.target_net.predict(np.reshape(next_states, [1, self.n_obs]))
            else:
                next_target = self.policy_net.predict(np.reshape(next_states, [1, self.n_obs]))

            next_q_value = self.getQvalue(rewards, next_target, dones)

            X_batch = np.append(X_batch, np.array([states]), axis=0)
            Y_sample = q_value.copy()

            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            if dones:
                X_batch = np.append(X_batch, np.array([next_states]), axis=0)
                Y_batch = np.append(Y_batch, np.array([[rewards] * self.n_actions]), axis=0)

        history = self.policy_net.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)
        return history.history['loss'][0]

    def policy_optimize(self, isStable = False):
        if len(self.memory) < self.batch_size:
            return -1
        return self.trainModel(isStable)
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, self.batch_size)
        i =0 
        for state, action, reward, next_state, done in minibatch:
            # y_target = self.target_net.predict(state)[0] # (shape = (BATCH_SIZE, n_actions) = (1, n_actions))
            # q_value = self.target_net.predict(state)[0] # WRONG!!!
            q_value = self.policy_net.predict(state)[0] # (shape = (batch_size, n_actions) = (1, n_actions))
            target_q_value = self.target_net.predict(next_state)[0] if isStable else self.policy_net.predict(next_state)[0]
            # update reward action `y_target[action]` that we choose, otherwise we keep the old reward
            q_value[action] = reward if done else reward + \
                self.gamma * np.max(target_q_value)
            x_batch.append(state[0].copy()) # since we batch, we concatenate state.shape=(1, x_size) state with axis=0
            y_batch.append(q_value.copy())
            if done:
                x_batch.append(next_state[0].copy())
                y_batch.append([reward] * self.n_actions)
        history = self.policy_net.fit(np.array(x_batch), np.array(y_batch),
                       batch_size=self.batch_size, epochs=1, verbose=0)
        return history.history['loss'][0]
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

    def run(self, num_episodes, do_train=True):
        reward_history= np.array([])
        reward_averaged = np.array([])
        reward_acc = 0
        loss_episode =[]
        global_step = 0
        start_time = time.time()
        first_time = True
        info_msg = Float32MultiArray()
        # state_msg = Float32MultiArray()
        for episode_count in range(1 + self.load_episode, num_episodes):
            state = self.env.reset()
            done = False
            reward_acc = 0
            for t in count():
                # openai_ros doesnt support render for the moment
                # eps= EPS_END + (EPS_START - EPS_END) * \
                #     math.exp(-1. * episode_count / EPS_DECAY)

                action = self.select_action(state, self.epsilon)

                next_state, reward, done, _ = self.env.step(action)
                reward_acc += reward
                info_msg.data = [action, reward, reward_acc]
                self.pub_info.publish(info_msg)
                # state_msg.data = state
                # self.pub_state.publish(state_msg)

                if do_train:
                    # If we are training we want to remember what I did and process it.
                    self.remember(state, action, reward, next_state, done)
                    if global_step >= START_USE_TARGET_STEPS:
                        loss_episode.append(self.policy_optimize(isStable = True))
                    else:
                        loss_episode.append(self.policy_optimize(isStable= False))
                state = next_state
                
                if t >= 500:
                    print("Time out!!")
                    done = True
                if done:
                    self.target_net.set_weights(self.policy_net.get_weights())
                    reward_history = np.append(reward_history, reward_acc)
                    reward_averaged = np.append(reward_averaged, np.average(np.array(reward_history[-50:])))
                    m, s = divmod(int(time.time() - start_time), 60)
                    h, m = divmod(m, 60)
                    print("[episode:{}/{}] score: {} best:{} avg:{:.4f} avg_loss {} eps {} memory {} time: {}: {:02}:{:02}".format(
                        episode_count, num_episodes, reward_acc, np.max(reward_history),
                        np.mean(reward_history[-10:]), np.mean(loss_episode[-10:]), self.epsilon, len(self.memory), h, m, s))
                    loss_episode = []
                    param_keys = ['epsilon']
                    param_values = [self.epsilon]
                    param_dictionary = dict(zip(param_keys, param_values))
                    if episode_count % 10 == 0:
                        # self.save(self.dirPath + str(episode_count) + '.h5')
                        self.save("dqn" + str(episode_count), self.dirPath)
                        with open(self.dirPath + "dqn" +str(episode_count) + "_eps" + '.json', 'w') as outfile:
                            json.dump(param_dictionary, outfile)
                    break
                global_step += 1
                if first_time and global_step >= START_USE_TARGET_STEPS:
                    print("UPDATE TARGET NETWORK")
                    first_time = False
            if self.epsilon > EPS_END:
                self.epsilon *= EPS_DECAY

        return episode_count
    

    def save(self, model_name, models_dir_path="/models"):
        """
        We save the current model
        """
        
        model_name_yaml_format = model_name+".yaml"
        model_name_HDF5_format = model_name+".h5"
        
        model_name_yaml_format_path = os.path.join(models_dir_path,model_name_yaml_format)
        model_name_HDF5_format_path = os.path.join(models_dir_path,model_name_HDF5_format)
        
        # serialize model to YAML
        model_yaml = self.policy_net.to_yaml()
        
        with open(model_name_yaml_format_path, "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5: http://www.h5py.org/
        self.policy_net.save_weights(model_name_HDF5_format_path)
        print("Saved model to disk")
        
    def load(self, model_name, models_dir_path="/models"):
        """
        Loads a previously saved model
        """
        
        model_name_yaml_format = model_name+".yaml"
        model_name_HDF5_format = model_name+".h5"
        
        model_name_yaml_format_path = os.path.join(models_dir_path,model_name_yaml_format)
        model_name_HDF5_format_path = os.path.join(models_dir_path,model_name_HDF5_format)
        
        # load yaml and create model
        yaml_file = open(model_name_yaml_format_path, 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        self.policy_net = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        self.policy_net.load_weights(model_name_HDF5_format_path)
        self.policy_net.compile(loss='mse', optimizer=RMSprop(lr=self.alpha, rho=0.9, epsilon=1e-06))
        print("Loaded model from disk")