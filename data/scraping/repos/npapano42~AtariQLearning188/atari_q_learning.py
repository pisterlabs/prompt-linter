import numpy as np
import gym
import tensorflow as tf
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from collections import deque, Counter
from tensorflow.contrib.layers import flatten, conv2d, fully_connected
import random
from datetime import datetime
from gym import wrappers
from time import time


# initialize variables
RGB = np.array([210, 164, 74]).mean()
STEPS_PER_EPSILON = 500000
DELAY = 20000
EPISODES = 1200
BATCH_SIZE = 50
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.97
INPUT_SHAPE = (None, 88, 80, 1)
X_SHAPE = (None, 88, 80, 1)

# Epsilon value and thresholds
EPSILON = 0.5
MIN_EPSILON = 0.05
MAX_EPSILON = 1

# Variables for training iterations
TRAINING_STEPS = 4
START_STEPS = 2000
COPY_STEPS = 100

exp_buffer = deque(maxlen=DELAY)
global_step = 0


def preprocess_image(obs):
    """Reshape game input from OpenAI gym to fit model input shape

    Parameters:
    obs -- The current frame of the video game from OpenAI Gym
    """
    res = obs[1:176:2, ::2]
    res = res.mean(axis=2)
    res[res == RGB] = 0
    res = (res-128)/128 - 1
    return res.reshape(88, 80, 1)


def dqn(x, scope, n_outputs):
    """Generate Deep Q Network.

    Parameters:
    x -- Random value populated empty Q table
    scope -- Type of table to generate, either mainQ or targetQ
    n_outputs -- number of possible actions to take
    """
    initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(scope) as cur_scope:
        layer1 = conv2d(x, num_outputs=32, kernel_size=(
            8, 8), stride=4, padding='SAME', weights_initializer=initializer)
        tf.summary.histogram('layer1', layer1)

        layer2 = conv2d(layer1, num_outputs=64, kernel_size=(
            4, 4), stride=2, padding='SAME', weights_initializer=initializer)
        tf.summary.histogram('layer2', layer2)

        layer3 = conv2d(layer2, num_outputs=64, kernel_size=(
            3, 3), stride=1, padding='SAME', weights_initializer=initializer)
        tf.summary.histogram('layer3', layer3)

        flat = flatten(layer3)

        fully_con = fully_connected(
            flat, num_outputs=128, weights_initializer=initializer)
        tf.summary.histogram('fully_con', fully_con)

        output = fully_connected(fully_con, num_outputs=n_outputs,
                                 activation_fn=None, weights_initializer=initializer)
        tf.summary.histogram('output', output)

        params = {v.name[len(cur_scope.name):]: v for v in tf.get_collection(
            key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=cur_scope.name)}
        return params, output


def get_sample(batch_size):
    """Generate sample output for actions and reward calculation.

    Parameters:
    batch_size -- size of table output
    """
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    mem = np.array(exp_buffer)[perm_batch]
    return mem[:, 0], mem[:, 1], mem[:, 2], mem[:, 3], mem[:, 4]


def epsilon_greedy(action, step):
    """Generate suggestied action using greedy epsion algorithm.

    Parameters:
    action -- The default action to be made.
    step -- Helper variable to calculate epsilon value.
    """
    epsilon = max(MIN_EPSILON, MAX_EPSILON -
                  (MAX_EPSILON - MIN_EPSILON) * step/STEPS_PER_EPSILON)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        return action


env = gym.make("Breakout-v0")
n_outputs = env.action_space.n
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=X_SHAPE)
y = tf.placeholder(tf.float32, shape=(None, 1))
in_training_mode = tf.placeholder(tf.bool)

main_dqn, main_dqn_outputs = dqn(x, 'mainQ', n_outputs)
target_dqn, target_dqn_outputs = dqn(x, 'targetQ', n_outputs)

x_action = tf.placeholder(tf.int32, shape=(None,))
q_action = tf.reduce_sum(
    target_dqn_outputs * tf.one_hot(x_action, n_outputs), axis=-1, keep_dims=True)

copy_operation = [tf.assign(main_name, target_dqn[var_name])
                  for var_name, main_name in main_dqn.items()]
copy_target_to_main = tf.group(*copy_operation)

loss = tf.reduce_mean(tf.square(y-q_action))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
training_operation = optimizer.minimize(loss)

init = tf.global_variables_initializer()

loss_summary = tf.summary.scalar('LOSS', loss)
merge_summary = tf.summary.merge_all()

with tf.Session() as session:
    init.run()

    for i in range(EPISODES):
        done = False
        obs = env.reset()
        env.render()
        epoch = 0
        episodic_reward = 0
        actions_counter = Counter()
        episodic_loss = []

        print(i)

        while not done:
            
            env.render()
            obs = preprocess_image(obs)
            actions = main_dqn_outputs.eval(
                feed_dict={x: [obs], in_training_mode: False})
            action = np.argmax(actions, axis=-1)
            actions_counter[str(action)] += 1
            action = epsilon_greedy(action, global_step)
            next_obs, reward, done, _ = env.step(action)
            exp_buffer.append(
                [obs, action, preprocess_image(next_obs), reward, done])

            if global_step % TRAINING_STEPS == 0 and global_step > START_STEPS:
                o_obs, o_act, o_next_obs, o_reward, o_done = get_sample(
                    BATCH_SIZE)
                o_obs = [x for x in o_obs]
                o_next_obs = [x for x in o_next_obs]
                next_act = main_dqn_outputs.eval(
                    feed_dict={x: o_next_obs, in_training_mode: False})
                y_batch = o_reward + DISCOUNT_FACTOR * \
                    np.max(next_act, axis=-1) * (1-o_done)

                train_loss, _ = session.run([loss, training_operation], feed_dict={x: o_obs, y: np.expand_dims(y_batch, axis=-1),
                                                                                   x_action: o_act, in_training_mode: True})
                episodic_loss.append(train_loss)

            if (global_step+1) % COPY_STEPS == 0 and global_step > START_STEPS:
                copy_target_to_main.run()

            obs = next_obs
            epoch += 1
            global_step += 1
            episodic_reward += reward
