import gym
import gym_gomoku
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time

'''
NOTE: Arrays are row major, i.e.
2 x 3:
    1, 2, 3
    4, 5, 6
'''

env = gym.make('Gomoku19x19-v0')

discount_factor = 0.9

'''
Discounted Rewards:
    Given the entire history of rewards for an episode, we want to work backwards to build an array of discounted rewards.
    Given a discount rate = d
    The reward at a time T (r.T) will be equal to r.T + (r.T+1) + d*r.T+2 + (d^2)*r.T+3 + ...
'''
def discount_rewards(r):
    # Create an array of zeros the same dimensions as r
    discounted_r = np.zeros_like(r)
    running_sum = 0

    # Start from the back of the list and work forward
    for t in reversed(range(0, r.size)):
        running_sum = running_sum * discount_factor + r[t]
        discounted_r[t] = running_sum
    return discounted_r

class agent():
    
    def __init__(self, learning_rate, state_size, action_size, hidden_layer_size):

        '''
        Neural Network:
            Input Layer: For Gomoku will be a 1 x 361 tensor, a flat representation of 19 x 19 board
            Hidden Layer: Fully connected to the input, using Sigmoid as the activation function
            Output Layer: For Gomoku will be a 1 x 361 tensor, a flat representation of 19 x 19 board

            Chosen Action: Which ever spot on the board has the highest value
        '''
        # This established the feedforward part of the network
        self.state_in = tf.placeholder(shape = [None, state_size], dtype = tf.float32)
        hidden = slim.fully_connected(self.state_in, hidden_layer_size, activation_fn = tf.nn.relu)
        self.output = slim.fully_connected(hidden, action_size, activation_fn = tf.nn.softmax)
        self.chosen_action = tf.argmax(self.output, 1)

        self.reward_var = tf.placeholder(shape = [None], dtype = tf.float32)
        self.action_var = tf.placeholder(shape = [None], dtype = tf.int32)

        # Backpropogation
        '''
        Backpropogation:
            Indexes: This line seems to be convaluted (future proof?) way to get the index of the chosen action
                It appears to be 0 * 361 + action_var = action_var
            Responsible Outputs: Reshapes output to be flat, then gathers the params from output corresponding
                to the indecies we provide, which is just the one action
            Loss: Reduce the loss or seomthing like that
        '''
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_var
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_var)

        # Not too sure what this does below, but I think it has to do with gradient descent and mini-batches
        trainable_vars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(trainable_vars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + "_holder")
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, trainable_vars)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, trainable_vars))

# Cleanup for TensorFlow
tf.reset_default_graph()

myAgent = agent(learning_rate = 0.7, state_size = 361, action_size = 361, hidden_layer_size = 45)

total_episodes = 50000
max_moves = 361
update_frequency = 5 # Number of episodes until update the network (batch gradient descent?)
exploratory_chance = 0.10
number_wins = 0

init = tf.global_variables_initializer()

# Launch the TensorFlow Graph
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_reward = []
    total_length = []

    gradientBuffer = sess.run(tf.trainable_variables())
    for idx, gradient in enumerate(gradientBuffer):
        gradientBuffer[idx] = gradient * 0

    random_moves = 0
    invalid_moves = 0
    network_moves = 0

    #while i < total_episodes:
    while True:
        state = env.reset()
        running_reward = 0
        ep_history = []
         
        for j in range(max_moves):
            # Get an action from the network
            state = np.reshape(state, [-1]).astype(float)
            for idx in range(len(state)):
                if state[idx] >= 2:
                    state[idx] = -1

            a = sess.run(myAgent.chosen_action, feed_dict = {myAgent.state_in:[state]})
            action = a[0]

            # Given a certain probability, make random exploratory moves
            random_num = np.random.rand(1)
            if random_num < exploratory_chance:
#                print("Made a random move!")
                random_moves += 1
                action = env.action_space.sample()
            elif action not in env.action_space.valid_spaces:
#                print("Move from network was not valid! " + str(action))
                invalid_moves += 1
                action = env.action_space.sample()
            else:
                network_moves += 1
#                print("Took a move from the network")

            # The state we receive from OpenAI Gym is a 19x19 grid with 1 and 2 to represent the players
            # Need to flatten the tensor into 1 x 361 and then scale the values to -1 and 1
            state_prime, reward, done, _ = env.step(action)
            ep_history.append([state, action, reward, state_prime])
            state = state_prime
            running_reward += reward

            if done == True:
                if reward == 1:
                    number_wins += 1
                    print("Player wins! Episode: " + str(i))
                    env.render()
                    time.sleep(5)

                # Update the network
                ep_history = np.array(ep_history)
                ep_history[:,2] = discount_rewards(ep_history[:,2]) # Take the all of the rewards from episode history and discount them
                feed_dict = {myAgent.reward_var:ep_history[:,2],
                             myAgent.action_var:ep_history[:,1],
                             myAgent.state_in:np.vstack(ep_history[:,0])}
                grads = sess.run(myAgent.gradients, feed_dict = feed_dict)
                for idx, gradient in enumerate(grads):
                    gradientBuffer[idx] += gradient

                if i % update_frequency == 0 and i != 0:
                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradientBuffer))
                    _ = sess.run(myAgent.update_batch, feed_dict = feed_dict)
                    for ix, grad in enumerate(gradientBuffer):
                        gradientBuffer[ix] = grad * 0

                total_reward.append(running_reward)
                total_length.append(j)
                break
        if i % 100 == 0:
            print("Episodes completed: " + str(i) + " " + str(np.mean(total_reward[-100:])))
            print("Number of wins so far: " + str(number_wins))
            print("\tRandom moves: " + str(random_moves))
            print("\tNetwork moves: " + str(network_moves))
            print("\tInvalid moves: " + str(invalid_moves))
            print("\n\tTotal moves: " + str(network_moves + invalid_moves + random_moves))
            print("\n\tAvg moves per game: " + str((network_moves + invalid_moves + random_moves) / 100))

            env.render()
            
            file = open("running_data.txt", "a")
            file.write("\n\nEpisodes completed: " + str(i) + " " + str(np.mean(total_reward[-100:])))
            file.write("\nNumber of wins so far: " + str(number_wins))
            file.write("\n\tRandom moves: " + str(random_moves))
            file.write("\n\tNetwork moves: " + str(network_moves))
            file.write("\n\tInvalid moves: " + str(invalid_moves))
            file.write("\n\tTotal moves: " + str(network_moves + invalid_moves + random_moves))
            file.write("\n\tAvg moves per game: " + str((network_moves + invalid_moves + random_moves) / 100))
            file.close()
            
            random_moves = 0
            network_moves = 0
            invalid_moves = 0

        i += 1


