import random
import gym
import numpy as np
from DQL_agents import DQLAgent
from QL_agents_preprocessing import BreakoutPreprocessor
from DQL_agents_preprocessing import preprocess, get_next_state
import h5py

# Constants
MAX_EPISODES = 1000
MAX_EPISODE_STATES = 10000
MAX_STATES = 200000  # 200000

# Create file
hf = h5py.File('/Users/s1500625/Desktop/data_preprocessor_breakout.h5', 'w')
hf.create_dataset('X_train', (MAX_STATES, 210, 160, 3), np.uint8)
hf.create_dataset('Y_train', (MAX_STATES, 740), np.uint8)  # (160+210)*2

# Initialize preprocessor
p = BreakoutPreprocessor()

# Get data from OpenAI
env = gym.make('BreakoutDeterministic-v4')

# Define agent
agent = DQLAgent('breakout')
agent.model.load_weights('DQN_breakout_weights12000000.hdf5')

frame_counter = 0

# Generate random indexes to shuffle database
random_indexes = np.arange(MAX_STATES)
np.random.shuffle(random_indexes)

for episode in xrange(MAX_EPISODES):
    raw_obs = env.reset()
    obs = preprocess(raw_obs)
    # Initialize the first state with the same 4 images
    current_state = np.array([[obs, obs, obs, obs]], dtype=np.uint8).reshape((105, 80, 4))

    for t in xrange(MAX_EPISODE_STATES):
        # run environment
        # env.render()

        # Choose the action according to the behaviour policy
        if random.random() < 0.4:
            action_index = env.action_space.sample()
        else:
            action_index = agent.choose_best_action(current_state)

        # Play one game iteration
        raw_obs, reward, is_done, _ = env.step(action_index)
        obs = preprocess(raw_obs)
        next_state = get_next_state(current_state, obs)

        # make next_state the new current state for the next frame.
        current_state = next_state

        # random index to shuffle the dataset
        rand_indx = random_indexes[frame_counter]

        if is_done:
            # print the score and break out of the loop
            print("episode: {}, episode_states: {}, tot_states: {}"
                  .format(episode, t, frame_counter))
            break

        if frame_counter < MAX_STATES:
            # write X
            hf['X_train'][rand_indx, :] = raw_obs

            # get features and save to Y
            center_features = p.get_y(raw_obs)
            hf['Y_train'][rand_indx, :] = center_features
        else:
            # close
            hf.close()
            break

        frame_counter += 1

# open and read
hf = h5py.File('/Users/s1500625/Desktop/data_preprocessor_breakout.h5', 'r')

x1 = hf.get('X_train')
x1 = np.array(x1[-1, :])  # index like a np array 8)

y1 = hf.get('Y_train')
y1 = np.array(y1[-1, :])

print x1.shape
print x1
print ""
print y1.shape
print y1
hf.close()
