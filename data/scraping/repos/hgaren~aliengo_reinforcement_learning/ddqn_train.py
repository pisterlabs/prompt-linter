
import os
import random
import gym
import pylab
import numpy as np
import time
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
import matplotlib.pyplot as plt

# Openai and ROS packages 
from openai_ros.openai_ros_common import StartOpenAI_ROS_Environment
from openai_ros.task_envs.task_commons import LoadYamlFileParamsTest
import rospy
import rospkg
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelConfiguration




rospy.init_node('ddqn_aliengo_train_node')
# True if double dqn , False Standradnt dqn
ddqn = False

def OurModel(input_shape, action_space):
    X_input = Input(input_shape)
    X = X_input

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(100, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 256 nodes
    X = Dense(100, activation="relu", kernel_initializer='he_uniform')(X)
    
    # Hidden layer with 64 nodes
    X = Dense(100, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X)
    model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model

class DQNAgent:
    def __init__(self):
       
        self.state_size = 3
        self.action_size = 8

        self.EPISODES = 100000
        self.memory = deque(maxlen=2000)
        
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.batch_size = 32
        self.train_start = 1000


        self.Soft_Update = False

        self.TAU = 0.1 # target network soft update hyperparameter

        self.episodes, self.average =  [], []
        
        # create main model
        self.model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)
        self.target_model = OurModel(input_shape=(self.state_size,), action_space = self.action_size)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        if not self.Soft_Update and ddqn:
            self.target_model.set_weights(self.model.get_weights())
            return
        if self.Soft_Update and ddqn:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1-self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(self.batch_size, self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)
        target_val = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if ddqn: # Double - DQN
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])   
                else: # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)
 
    def run(self):
        # Initialize OpenAI_ROS ENV
        LoadYamlFileParamsTest(rospackage_name="learning_ros", rel_path_from_package_to_file="config", yaml_file_name="aliengo_stand.yaml")
        env = StartOpenAI_ROS_Environment('AliengoStand-v0')
        saver = tf.train.Saver()

        time.sleep(3)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            scores = []
            success_num = 0

            for e in range(self.EPISODES):
                state = env.reset()
                state = np.reshape(state, [1, self.state_size])
                done = False
                rewards = []
                while  not rospy.is_shutdown():  # until ros is not shutdown                #self.env.render()
                    action = self.act(state)
                    next_state, reward, done, _ = env.step(action)
                    time.sleep(0.01)

                    next_state = np.reshape(next_state, [1, self.state_size])
                    rewards.append(reward)
                    self.remember(state, action, reward, next_state, done)
                    
                    if done:
                        # every step update target model
                        self.update_target_model()
                        

                        state = env.reset()
                        break 
                    else:
                        state = next_state   
                    self.replay()
                      #if consectuvely 10 times has high reward end the training
                scores.append(sum(rewards))
                if sum(rewards) >=400:
                    success_num += 1
                    print("Succes number: " + str(success_num))
                    if success_num >= 5: #checkpoint
                        if(ddqn):
                            saver.save(sess, 'model_train/model_ddqn.ckpt')
                        else:
                            saver.save(sess, 'model_train/model_dqn.ckpt')

                        print('Clear!! Model saved.')
                    if success_num >= 10:
                        if(ddqn):
                            saver.save(sess, 'model_train/model_ddqn.ckpt')
                        else:
                            saver.save(sess, 'model_train/model_dqn.ckpt')                       

                        print('Clear!! Model saved. AND Finished! ')
                        break
        
                else:
                    success_num = 0
        plt.plot(scores)
        plt.show()        
if __name__ == "__main__":
    agent = DQNAgent()
    agent.run()
