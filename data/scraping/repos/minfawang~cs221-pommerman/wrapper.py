import pommerman
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme
from pommerman.agents import SimpleAgent, BaseAgent
from tensorforce.contrib.openai_gym import OpenAIGym
from datetime import datetime
from pommerman import utility
import copy
import io
import random
import json
import os
import tempfile


DEBUG = False
SAMPLE_FROM_LAST_K_STATES = 10
MAX_NUM_STATES_TO_TRAIN_PER_EPISODE = 100
LOST_TIE_DROP_PROB = 0.75

def dprint(*args, **kwargs):
  if DEBUG:
    print(*args, **kwargs)


def make_demo_gym():
  # Add 4 random agents
  config = ffa_v0_fast_env()
  agents = [
      SimpleAgent(config["agent"](agent_id, config["game_type"]))
      for agent_id in range(4)
  ]

  # return pommerman.make('PommeFFACompetition-v0', agents)
  # Instantiate the environment
  env = Pomme(**config["env_kwargs"])
  # env.seed(0)

  env.set_agents(agents)
  # env.set_training_agent(agents[-1].agent_id)
  env.set_init_game_state(None)

  return env


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


class BackPlayWrappedEnv(OpenAIGym):
    """
    This wrapper enables BackPlay by storing 2 gym envs:
    * The "gym" is the main learning environment. Its initial state is derived
      from the "demo_gym".
    * The "demo_gym" is going to play a full game without learning. It will
      take snapshots of every single state of the game, and then feeds a subset
      of the states to the "gym" as its initial learning state.

    We down-sample the number of states to be fewer than or equal to
    MAX_NUM_STATES_TO_TRAIN_PER_EPISODE. It ensures the model to learn more
    efficiently by exploring more variantions of situations.

    In the sub-sampled states, we repeat the following procedures until the
    remaining states if empty:
      1. Sample a state from the last SAMPLE_FROM_LAST_K_STATES states.
      2. Remove that state from the states and use it as the inital state for
         "gym".
      3. Train the agent in "gym".
    """
    def __init__(self, gym, agent_reward_fn, featurize_fn,
                 state_root_dir, episode_number=0, train_wins_only=False, visualize=False):
        self.demo_gym = make_demo_gym()
        self.gym = gym
        self.state_root_dir = state_root_dir
        self.visualize = visualize
        self.agent_reward_fn_ = agent_reward_fn
        self.featurize_fn_ = featurize_fn
        self.demo_states = None
        self.train_wins_only = train_wins_only

        if not os.path.exists(state_root_dir):
          os.makedirs(state_root_dir)

        self.episode_number = episode_number
        # Check if there is some snapshot that is above the episode number,
        # which can happen if we terminate the program in the middle of training.
        max_episode_from_state_dir = max(
            [int(episode) for episode in os.listdir(state_root_dir)] or [0])
        if episode_number + 1 <= max_episode_from_state_dir:
          print('[WARNING] episode number ({}) < max_episode_from_state_dir ({}). Overwritting episode number...'.format(
              episode_number, max_episode_from_state_dir))
          self.episode_number = max_episode_from_state_dir

    def get_sorted_game_states(self, game_state_dir):
      def is_alive(agent_id, agents):
        for agent in json.loads(agents):
          if agent['agent_id'] == agent_id:
            return agent['is_alive']
        return False

      AGENT_ID = 3
      state_file = os.path.join(game_state_dir, 'game_state.json')
      with open(state_file, 'r') as fin:
        game_states = json.load(fin)

        is_tied = 'winners' not in game_states
        is_lost = (not is_tied) and (AGENT_ID not in game_states['winners'])
        # Skip this episode if it's configured to train on winning games only.
        if self.train_wins_only and (is_tied or is_lost):
          return []

        states = sorted(
            [state for state in game_states['state'] if is_alive(AGENT_ID, state['agents'])],
            key=lambda state: int(state['step_count']),
        )

        # Keep down-sampling the states by half until its size is below the ceiling.
        while len(states) > MAX_NUM_STATES_TO_TRAIN_PER_EPISODE:
          states = [state for i, state in enumerate(states) if i % 2]

        if is_tied or is_lost:
          if random.random() < LOST_TIE_DROP_PROB:
            # Drop all states and only keep the initial state.
            return states[:1]

        return states

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
        agent_state = self.featurize_fn_(state[self.gym.training_agent])
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

    def record_demo(self):
      self.episode_number += 1
      os.makedirs(os.path.join(self.state_root_dir, str(self.episode_number)))

      config = 'PommeFFACompetition-v0'
      _agents = ['test::agents.SimpleAgent'] * 4
      env = self.demo_gym
      obs = env.reset()
      done = False
      record_json_dir = os.path.join(self.state_root_dir, str(self.episode_number))

      while not done:
        if self.visualize:
          env.render(
              record_json_dir=record_json_dir,
              do_sleep=False)
        else:
          env.save_json(record_json_dir)
        actions = env.act(obs)
        obs, reward, done, info = env.step(actions)

      if self.visualize:
        env.render(
            record_json_dir=record_json_dir,
            do_sleep=False)
      else:
        env.save_json(record_json_dir)
      env.render(close=True)

      finished_at = datetime.now().isoformat()
      utility.join_json_state(
          record_json_dir, _agents, finished_at, config, info)

      return record_json_dir

    def get_training_demo_states(self):
      def didAgentWin(game_state_dir):
        AGENT_ID = 3
        state_file = os.path.join(game_state_dir, 'game_state.json')
        with open(state_file, 'r') as fin:
          game_states = json.load(fin)
          return AGENT_ID in game_states.get('winners', [])

      found_record = False
      while not found_record:
        record_json_dir = self.record_demo()
        found_record = didAgentWin(record_json_dir) or (not self.train_wins_only)
      return self.get_sorted_game_states(record_json_dir)

    def get_next_demo_state(self):
      if not self.demo_states:
        self.demo_states = self.get_training_demo_states()
        if not self.demo_states:
          return None

      # Back play: Randomly pick one state from the last K states.
      sample_size = min(SAMPLE_FROM_LAST_K_STATES, len(self.demo_states))
      index = -1 - random.choice(range(sample_size))
      state = self.demo_states.pop(index)
      return state

    def update_init_game_state(self, state):
      fd, game_state_file = tempfile.mkstemp()
      try:
        with os.fdopen(fd, 'w') as tmp:
          tmp.write(json.dumps(state))

        self.gym.set_init_game_state(game_state_file)
      finally:
        os.remove(game_state_file)

    def reset(self):
        state = self.get_next_demo_state()
        self.update_init_game_state(state)

        obs = self.gym.reset()
        agent_obs = self.featurize_fn_(obs[3])
        return agent_obs
