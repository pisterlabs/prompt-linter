""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import cPickle as pickle
import gym
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Used to display W1 activations
def display_W1(model,H):
    firstLayer = model['W1']
    for neuron_idx in range(0,H):
        print('Display neuron %d' % neuron_idx)
        neuron = firstLayer[neuron_idx,:]
        neuron = neuron.reshape(-1,80)
        plt.imshow(neuron)
        neuronFileName = 'neuron_%d' % neuron_idx
        plt.imsave(neuronFileName, neuron, format='png')
        plt.show()

# Save pre-processed input image
def saveInput(I,conter):
    plt.imshow(I.reshape(80,80))
    imageFilename = 'inDiff_%d' % conter
    plt.imsave(imageFilename, I.reshape(80,80), format='png')


# hyperparameters
# number of hidden layer neurons
H = 200
# every how many episodes to do a param update?
batch_size = 10
learn_rate = 1e-4
# discount factor for reward
gamma = 0.99
# decay factor for RMSProp leaky sum of grad^2
decay_rate = 0.99
# resume from previous checkpoint?
resume = True
# Put to true to show game screen
showGame = False
# Save W1 weights
showWeights = False
# Save input images
saveInputImages = False

# model initialization
# The inputs will be a 80x80 difference image (current frame - last frame)
# W1 will be 200,(80x80=640), so we will be able to learn 200 different
# "states"
D = 80 * 80 # input dimensionality: 80x80 grid
if resume:
    # Load saved model
    print('Loading pre-trained model')
    model = pickle.load(open('save.p', 'rb'))

    if showWeights == True:
        print('Converting gradients to 80x80')
        display_W1(model,H)
else:
    model = {}
    # Do "Xavier" initialization W1=[200x640] W2=[200]
    # Our network have 200(H) neurons on the input layer and 1 output neuron
    model['W1'] = np.random.randn(H,D) / np.sqrt(D)
    model['W2'] = np.random.randn(H) / np.sqrt(H)

# update buffers that add up gradients over a batch
grad_buffer = { k : np.zeros_like(v) for k,v in model.iteritems() }
# rmsprop memory
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.iteritems() }

# Used at the end of neural network to convert scores to probability
def sigmoid(x):
    # sigmoid "squashing" function to interval [0,1]
    return 1.0 / (1.0 + np.exp(-x))

# Image preprocessing....
def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    # Convert to 1d(ravel) row-wise
    return I.astype(np.float).ravel()

# Gives a preference to fast rewards, by discounting old rewards values
def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        # reset the sum, since this was a game boundary (pong specific!)
        if r[t] != 0: running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# Normal neural network forward propagation (Policy network)
def policy_forward(x):
    # X,W1 dot product (Matrix multiplication)
    h = np.dot(model['W1'], x)

    # Apply ReLU nonlinearity
    h[h<0] = 0

    # h(after relu),W2 dot product (Matrix multiplication)
    logp = np.dot(model['W2'], h)

    # get probability by squashing result
    p = sigmoid(logp)

    # return probability of taking action 2, and hidden state
    return p, h

# Normal neural network backward propagation
def policy_backward(previous_w1_h, previous_x, dout):
    # Backward fully conected layer (W2)
    dW2 = np.dot(previous_w1_h.T, dout).ravel()

    # Backward Relu
    dh = np.outer(dout, model['W2'])
    dh[previous_w1_h <= 0] = 0

    # Backward fully conected layer (W1)
    dW1 = np.dot(dh.T, previous_x)

    # Return gradient for W1 and W2
    return {'W1':dW1, 'W2':dW2}

# Start pong from openAI gym
gameEnviroment = gym.make("Pong-v0")
observation = gameEnviroment.reset()

# Initialize variables
prev_x = None
inputs_vector,hidden_vector,error_vector,reward_vector = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
frameNumber = 0

# Game loop
while True:
    # Display game screen
    if showGame:
        gameEnviroment.render()

    # preprocess the observation(image current frame)
    cur_x = prepro(observation)

    # Calculate input with current and previous frame
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)

    # Used to debug input images
    if saveInputImages == True:
        saveInput(x,frameNumber)
    frameNumber = frameNumber + 1

    # Save previous frame
    prev_x = cur_x

    # forward the policy network with input image
    # Return action probability and hidden state activations
    action_UP_prob, h = policy_forward(x)

    # Action will be the atari joystick command check the available commands
    # here:
    # https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
    # np.random.uniform() return a non-gaussian number from 0..1
    # Here we sample our action(UP/DOWN) from our probability distribution
    action = 2 if np.random.uniform() < action_UP_prob else 5

    # Save input and hidden(W1) activations (needed later for backprop)
    inputs_vector.append(x) # input
    hidden_vector.append(h) # hidden state

    # y=1 if action==UP else 0 (Create label data), at this point we don't
    # know yet if the action was good/bad
    y = 1 if action == 2 else 0 # a "fake label"

    # Loss error used for regression problems
    # http://cs231n.github.io/neural-networks-2/#losses
    # At this point we don't know if this error was good (positive reward) or
    # bad(negative reward), we need to wait for the reward(-1,0,1) to modulate
    # this error and force the action to be executed or not
    error_vector.append(y - action_UP_prob)

    # step the environment and get measurements (including rewards)
    # Reward will be +1 if ball passed oponent size, -1 if passed your side
    # Reward and utility are the same thing
    observation, reward, done, info = gameEnviroment.step(action)
    reward_sum += reward

    # Save reward (call after "step" to get the reward of previous action)
    reward_vector.append(reward)

    # Game-over episode finished (Did we loose or win)
    # One complete episode means that the ball passed at least 20 times
    if done:
        episode_number += 1

        # stack together all inputs, hidden states, action gradients,
        # and rewards for this episode vertically (row-wise)
        batch_x = np.vstack(inputs_vector)
        batch_hidden_act = np.vstack(hidden_vector)
        batch_prob_grad = np.vstack(error_vector)
        batch_rewards = np.vstack(reward_vector)
        # reset arrays for next episodes
        inputs_vector,hidden_vector,error_vector,reward_vector = [],[],[],[]

        # compute the discounted reward backwards through time, this is done
        # to have preference to faster rewards, with this long rewards has less
        # utility...
        discounted_batch_rewards = discount_rewards(batch_rewards)

        # standardize the rewards to be unit normal
        discounted_batch_rewards -= np.mean(discounted_batch_rewards)
        discounted_batch_rewards /= np.std(discounted_batch_rewards)

        # This will modulate the gradient and make the current choice more/less
        # likely to happen on the future
        batch_prob_grad *= discounted_batch_rewards

        # Get network gradients (Backpropagation)
        grad = policy_backward(batch_hidden_act, batch_x, batch_prob_grad)
        for k in model:
            grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            print('********** Update gradients with rmsprop **************')
            for k,v in model.iteritems():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + \
                    (1 - decay_rate) * g**2
                model[k] += learn_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        # Consider the "runnig_reward" as out utility that must be maximized
        running_reward = reward_sum if running_reward is None else \
            running_reward * 0.99 + reward_sum * 0.01
        print 'resetting game. episode reward total was %f. running mean: %f' \
            % (reward_sum, running_reward)

        # Save model every 100 done episodes
        if episode_number % 100 == 0:
            print('Saving model....')
            pickle.dump(model, open('save.p', 'wb'))

        reward_sum = 0
        # Reset Atari...
        observation = gameEnviroment.reset() # reset env
        prev_x = None
        frameNumber = 0

    # Game finished (not the episode...) Possible rewards at this point (+1,-1)
    if reward != 0:
        print ('ep %d: game finished, reward: %f' % (episode_number, reward)) \
            + ('' if reward == -1 else ' !!!!!!!!')
