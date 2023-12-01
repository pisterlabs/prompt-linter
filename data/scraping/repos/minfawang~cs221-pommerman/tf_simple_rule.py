'''An example to show how to set up an pommerman game programmatically

Explanation of rewards:
# Suppose we have 4 agents, below may be a sequence of rewards after each step.
# Essentially the reward of an agent is: 0 if alive, -1 if dead, and +1 if
# you are the only one alive.
# reward = [0, 0, 0, 0]
# reward = [0, 0, 0, 0]
# reward = [0, 0, 0, -1]
# reward = [0, 0, -1, -1]
# reward = [-1, 1, -1, -1]

Run steps:
1) Create and update MODEL_DIR
2) Update other constants at the top:
   * SHOULD_RENDER: If true, will render the game.
   * NUM_EPISODES: Total training episodes.
   * REPORT_EVERY_ITER: Print metric info every these iters.
   * SAVE_EVERY_ITER: Save the model checkpoint to file every these iters.
'''
import pommerman
from pommerman.agents import SimpleAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme

import numpy as np
import collections
import copy
import os
import pickle
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym


from feature_extractor import featurize_map

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # This set the environment.

DEBUG = False
SHOULD_RENDER = False
NUM_EPISODES = 100000
MAX_EPISODE_TIMTESTAMPS = 2000
MODEL_DIR = '../model_dir/tf_simple_rule/'
REPORT_EVERY_ITER = 20
SAVE_EVERY_ITER = 100

reward_counter = collections.Counter()
episode_recorder = []


def get_win_loss(episode_recorder):
  rewards = [t[-1] for t in episode_recorder]
  wins = len([r for r in rewards if r > 0])
  losses = len(rewards) - wins
  return wins, losses


# Callback function printing episode statistics
def episode_finished(r):
  global reward_counter, episode_recorder
  reward = r.episode_rewards[-1]
  reward_counter[reward] += 1
  episode_recorder.append([r.episode, r.episode_timestep, reward])

  if r.episode % REPORT_EVERY_ITER == 0:
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(
          ep=r.episode, ts=r.episode_timestep, reward=reward))
    
    wins, losses = get_win_loss(episode_recorder[-REPORT_EVERY_ITER:])
    print('Episode win/loss: {}/{}, rate:{}'.format(wins, losses, float(wins)/losses))
    wins, losses = get_win_loss(episode_recorder)
    print('Overall win/loss: {}/{}, rate:{}'.format(wins, losses, float(wins)/losses))

  if r.episode % SAVE_EVERY_ITER == 0:
    print('saving model ...')
    r.agent.save_model(MODEL_DIR)

    print('reward_counter ...')
    with open(os.path.join(MODEL_DIR, 'reward_counter.p'), 'wb') as fout:
      pickle.dump(reward_counter, fout)

    print('episode recorder ...')
    with open(os.path.join(MODEL_DIR, 'episode_recorder.p'), 'wb') as fout:
      pickle.dump(episode_recorder, fout)

  return True


def dprint(*args, **kwargs):
  if DEBUG:
    print(*args, **kwargs)


# Debug logic
def get_actions_str(actions, agent_index):
    actions_map = {
        0: 'Stop',
        1: 'Up',
        2: 'Down',
        3: 'Left',
        4: 'Right',
        5: 'Bomb',
    }
    action_strs = [actions_map[a] for a in actions]
    action_strs[agent_index] = '[{}]'.format(action_strs[agent_index])
    return 'all_actions: {}'.format(action_strs)


# TODO(minfa): Update agent reward
# if terminal and agent_alive (a.k.a. agent_reward == 1): reward += 500
# if agent is killed: reward = -500 + 0.5 * t
# if agent is dead for a while:
def compute_agent_reward(old_state, state, game_reward):
  """
  Inputs:
      old_state: Raw agent old state.
      state: Raw agent state.
      game_reward: Raw game reward with respect to the agent.
  """
  reward = 0
  if game_reward == 1:
    reward = 1.0
  elif game_reward == -1:
    reward = -1.0
  else:
    old_num_alive = len(old_state['alive'])
    new_num_alive = len(state['alive'])
    dprint('alive old/new: ', old_num_alive, new_num_alive)
    if new_num_alive < old_num_alive:
      reward += 0.1

    # [6, 7, 8] are special weapons.
    # In the case below, the agent just eats a weapon.
    position = state['position']
    old_board_val = old_state['board'][position]
    dprint('old board is new board: ', old_state['board'] is state['board'])
    dprint('position: {}, old board val: {}'.format(position, old_board_val))
    if old_board_val in [6, 7, 8]:
      reward += 0.1

  return reward


def featurize(obs):
    feature_map = featurize_map(obs)
    return np.concatenate(list(feature_map.values()))

class TensorforceAgent(BaseAgent):
    def act(self, obs, action_space):
        pass


def create_env_agent():
  # Instantiate the environment
  config = ffa_v0_fast_env()
  env = Pomme(**config["env_kwargs"])
  env.seed(0)

  # Create a Proximal Policy Optimization agent
  agent = PPOAgent(
      states=dict(type='float', shape=(80)),
      actions=dict(type='int', num_actions=env.action_space.n),
      network=[
          dict(type='dense', size=64),
          dict(type='dense', size=64)
      ],
      batching_capacity=1000,
      step_optimizer=dict(
          type='adam',
          learning_rate=2e-4
      ),

      # PGModel
      baseline_mode='network',
      baseline=dict(type='custom', network=[
          dict(type='dense', size=64),
          dict(type='dense', size=64)
      ]),
      baseline_optimizer=dict(
          type='adam',
          learning_rate=2e-4
      ),
  )

  # Add 3 random agents
  agents = []
  for agent_id in range(3):
      agents.append(SimpleAgent(config["agent"](agent_id, config["game_type"])))

  # Add TensorforceAgent
  agent_id += 1
  agents.append(TensorforceAgent(config["agent"](agent_id, config["game_type"])))
  env.set_agents(agents)
  env.set_training_agent(agents[-1].agent_id)
  env.set_init_game_state(None)

  return (env, agent)


class WrappedEnv(OpenAIGym):
    def __init__(self, gym, agent_reward_fn, visualize=False):
        self.gym = gym
        self.visualize = visualize
        self.agent_reward_fn_ = agent_reward_fn

    def set_agent_reward_fn(agent_reward_fn):
      self.agent_reward_fn_ = agent_reward_fn

    def execute(self, action):
        if self.visualize:
            self.gym.render()

        actions = self.unflatten_action(action=action)

        obs = self.gym.get_observations()
        agent_old_state = copy.deepcopy(obs[self.gym.training_agent])

        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]

        agent_reward = self.agent_reward_fn_(
            agent_old_state,
            state[self.gym.training_agent],
            agent_reward)

        if DEBUG:
          print(get_actions_str(all_actions, self.gym.training_agent))
          print('agent_state, terminal, agent_reward: ',
              state[self.gym.training_agent], terminal, agent_reward)
          should_terminal = input('\n press any key to step forward (or "t" to terminal) \n')
          if should_terminal == 't':
            terminal = True

        return agent_state, terminal, agent_reward

    def reset(self):
        obs = self.gym.reset()
        agent_obs = featurize(obs[3])
        return agent_obs


def main():
  global reward_counter, episode_recorder

  # Print all possible environments in the Pommerman registry
  print(pommerman.REGISTRY)

  # Create a set of agents (exactly four)
  env, agent = create_env_agent()

  try:
    agent.restore_model(MODEL_DIR)
    with open(os.path.join(MODEL_DIR, 'reward_counter.p'), 'rb') as fin:
      reward_counter = pickle.load(fin)
    with open(os.path.join(MODEL_DIR, 'episode_recorder.p'), 'rb') as fin:
      episode_recorder = pickle.load(fin)
    print('Loaded model from episode: ', agent.episode)
    wins, losses = get_win_loss(episode_recorder)
    print('Past win/loss: {}/{}'.format(wins, losses))
  except Exception as e:
    print(e)
    print('training a new model')

  wrapped_env = WrappedEnv(
      env,
      visualize=SHOULD_RENDER,
      agent_reward_fn=compute_agent_reward,
  )
  runner = Runner(agent=agent, environment=wrapped_env)
  runner.run(
      episodes=NUM_EPISODES,
      max_episode_timesteps=MAX_EPISODE_TIMTESTAMPS,
      episode_finished=episode_finished,
  )

  try:
      runner.close()
  except AttributeError as e:
      print('AttributeError: ', e)


if __name__ == '__main__':
    main()
