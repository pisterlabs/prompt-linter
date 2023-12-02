'''Breakout-v0'''

'''Playing Cartpole-v0 using the deep-Q learning algorithm'''

# which game are we playing from OpenAI's gym?
ENV_NAME = "Breakout-v0"


loaded_model = "Breakout-v0-scratch_500000.json"
loaded_weights = "Breakout-v0-scratch_5000000.h5"
model_name = ENV_NAME + "-scratch_500000"
if loaded_model is not None:
    model_name = loaded_model
arch_name = model_name + ".json"

Internet = True  # connected to the internet (Slack)?
Watch = True  # watch the training
Train = False  # train or just play?
Record = False  # create video from testing?
observe = 10000  # how many steps to observe before training?
nb_steps = 500000  # number of steps
nb_test_episodes = 5  # number of test episodes
max_nb_steps = 50000  # either nb_test_episodes or max_nb_steps -- whatever is shorter
gamma = 0.99  # decay rate of past Observations
explore = 500000  # frames over which to anneal epsilon
final_epsilon = 0.01  # final value of epsilon
initial_epsilon = 0.2  # starting value of epsilon
memory_max = 50000  # number of previous transitions to remember
batch = 50  # size of minibatch

# dimensions of input image
img_rows = 84
img_cols = 84


print("Using Internet:", Internet)
print("Environment:", ENV_NAME)
print("Watching:", Watch)
print("Training:", Train)
print("Record:", Record)

# access locations file
exec(open("Locations.py").read())
print("locations successfully read.")


#import dependencies
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

import os
import sys
import random
import numpy as np
from collections import deque
import gym
from PIL import Image
from keras.models import Sequential, model_from_json
from keras.regularizers import l2
from keras.initializations import normal, uniform
from keras.optimizers import SGD, rmsprop, Adam
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense


# define various modules for reproducibility
def save_weights(which_model, filename):
    run = os.path.join(runs_dir, filename)
    which_model.save_weights(run, overwrite=True)
    print("weights saved as", filename)


def save_architecture(archname, string):
    arch = os.path.join(arch_dir, archname)
    open(arch, 'w').write(string)
    print("architecture saved as", archname)


# preprocess the images
def preprocess(color):
    x = skimage.color.rgb2gray(color)
    x = skimage.transform.resize(x, (img_rows, img_cols))
    x = skimage.exposure.rescale_intensity(x, out_range=(0, 255))
    x = x.reshape(1, x.shape[0], x.shape[1], 1)
    return x


# Get the environment and extract the number of actions
env = gym.make(ENV_NAME)
# np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.n


# construct the model
def construct_model():
    if loaded_model is not None:
        # load previous architecture
        model_name = loaded_model
        model_location = os.path.join(arch_dir, loaded_model)
        model = model_from_json(open(model_location).read())
        if loaded_weights is not None:
            # load previous weights
            weights_location = os.path.join(runs_dir, loaded_weights)
            model.load_weights(weights_location)

    else:
        model = Sequential()
        model.add(ZeroPadding2D((1, 1),
                                input_shape=(img_rows, img_cols, 1)))
        model.add(Convolution2D(nb_filter=32,  # number of filters
                                nb_row=3,  # number of rows in kernel
                                nb_col=3,
                                init='normal',
                                border_mode='same',  # padded with zeroes
                                W_regularizer=None,
                                activity_regularizer=None,
                                bias=True,
                                name='conv1'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3),
                               strides=(2, 2),
                               border_mode='valid'))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(32, 3, 3,
                                init='normal',
                                name='conv2'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3),
                               strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3,
                                init='normal',
                                name='conv3'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3),
                               strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(output_dim=512,
                        input_dim=True,
                        init='normal',  # initialize the weights
                        bias=True,
                        W_regularizer=l2(0.01)
                        ))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(output_dim=nb_actions,
                        init='normal'))
        model.add(Activation('linear'))

        json_string = model.to_json()
        save_architecture(arch_name, json_string)
    print("model constructed.")
    return model


# train the model
def train_model():
    # initialize values
    memory = deque()
    t = 1
    episode = 1
    status = "Observing"
    total_reward = 0
    rewards = []
    epsilon = initial_epsilon

    x_t_colored = env.reset()
    x_t = preprocess(x_t_colored)

    action_t0 = env.action_space.sample()
    x_t1_colored, reward, done, info = env.step(action_t0)
    x_t1 = preprocess(x_t1_colored)
    x = x_t1 - x_t

    # train the model
    while t < nb_steps:
        if Watch:
            env.render(mode='human')

        if random.random() <= epsilon:
            print("----------Random Action----------")
            action = env.action_space.sample()
        else:
            # Get the prediction of the Q-function
            q = model.predict(x)
            # choose action with highest prediction of the Q-function
            action = np.argmax(q)

        # run the selected action and observed next state and reward
        x_t = x_t1
        x_t1_colored, reward, done, info = env.step(action)
        total_reward += reward

        x_t1 = preprocess(x_t1_colored)
        x1 = x_t1 - x_t

        # store the transition in D
        memory.append((x, action, reward, x1, done))
        if len(memory) > memory_max:
            memory.popleft()

        # reduce epsilon gradually
        if epsilon > final_epsilon:
            epsilon -= (initial_epsilon - final_epsilon) / explore

        # update values
        x = x1

        if done:

            rewards.append(total_reward)
            print("********** EPISODE ENDED ***********")
            print("TOTAL REWARD:", total_reward,
                  "/EPISODE AVERAGE:", np.mean(rewards))

            # train on minibatches. We only train after observation period is
            # over so that there are sufficient number of frames to sample from
            if t > observe:
                # sample a minibatch to train on
                minibatch = random.sample(memory, batch)

                # 50, img_rows, img_cols, 1
                inputs = np.zeros(
                    (batch, x1.shape[1], x1.shape[2], x1.shape[3]))
                targets = np.zeros((batch, nb_actions))  # 50, 6

                # Now we do the experience replay
                for i in range(0, len(minibatch)):
                    state_t = minibatch[i][0]
                    action_t = minibatch[i][1]
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    done = minibatch[i][4]
                    inputs[i:i + 1] = state_t
                    targets[i] = model.predict(state_t)
                    Q_sa = model.predict(state_t1)
                    max_Q = np.max(Q_sa)

                    if done:
                        targets[i, action_t] = reward_t
                    else:
                        targets[i, action_t] = reward_t + gamma * max_Q

                loss = 0
                loss += model.train_on_batch(inputs, targets)

                print("Minibatch complete. Loss: ", loss)
                print("saving weights.")
                save_weights(model, model_name + ".h5")

            total_reward = 0
            episode += 1
            x_t_colored = env.reset()
            x_t = preprocess(x_t_colored)

            action_t0 = env.action_space.sample()
            x_t1_colored, reward, done, info = env.step(action_t0)
            x_t1 = preprocess(x_t1_colored)
            x = x_t1 - x_t

            if t > observe:
                status = "Training"

        print("TIMESTEP", t, "/ EPISODE", episode, "/ STATUS", status,
              "/ EPSILON", round(epsilon, 4), "/ ACTION", action, "/ REWARD", reward, "/ DONE", done)

        # update values
        t = t + 1

    if Internet:
        # send message to slack when finished
        slck = os.path.join(slack_dir, "Slack.py")
        exec(open(slck).read())
        print("message sent to Slack")


# test the model
def test_model():
    # initialize values
    t = 1
    episode = 1
    status = "Testing"
    total_reward = 0
    epsilon = initial_epsilon
    rewards = []

    x_t_colored = env.reset()
    x_t = preprocess(x_t_colored)

    action_t0 = env.action_space.sample()
    x_t1_colored, reward, done, info = env.step(action_t0)
    x_t1 = preprocess(x_t1_colored)
    x = x_t1 - x_t

    # train the model
    while t < max_nb_steps and episode < nb_test_episodes:
        if Watch:
            env.render(mode='human')

        if random.random() <= epsilon:
            print("----------Random Action----------")
            action = env.action_space.sample()
        else:
            # Get the prediction of the Q-function
            q = model.predict(x)
            # choose action with highest prediction of the Q-function
            action = np.argmax(q)

        # run the selected action and observed next state and reward
        x_t = x_t1
        x_t1_colored, reward, done, info = env.step(action)
        total_reward += reward

        x_t1 = preprocess(x_t1_colored)
        x1 = x_t1 - x_t

        # update values
        x = x1

        if done:
            rewards.append(total_reward)
            print("********** EPISODE ENDED ***********")
            print("TOTAL REWARD:", total_reward,
                  "/EPISODE AVERAGE:", np.mean(rewards))

            total_reward = 0
            episode += 1
            x_t_colored = env.reset()
            x_t = preprocess(x_t_colored)

            action_t0 = env.action_space.sample()
            x_t1_colored, reward, done, info = env.step(action_t0)
            x_t1 = preprocess(x_t1_colored)
            x = x_t1 - x_t

        print("TIMESTEP", t, "/ EPISODE", episode, "/ STATUS", status,
              "/ ACTION", action, "/ REWARD", reward, "/ DONE", done)

        # update values
        t = t + 1


# construct model
model = construct_model()
# compile the model
model.compile(loss='mse', optimizer='rmsprop')


if Train:
    train_model()
if not Train:
    test_model()
