import gym
import numpy as np
import logging

from openai_gym.actor_critic_agent import ActorCriticAgent


class GameNormalizer(object):
    discrete_action_space = np.array([-1, -0.5, -0.2, 0, 0.2, 0.5, 1.0])

    def normalize_action(self, action):
        return action / 2.0

    def denormalize_action(self, norm_action):
        return norm_action * 2.0

    def make_discrete_action(self, action):
        norm_action = self.normalize_action(action)
        action_idx = (np.abs(GameNormalizer.discrete_action_space - norm_action)).argmin()
        return action_idx

    def make_continuous_action(self, action_idx):
        norm_action = GameNormalizer.discrete_action_space[action_idx]
        action = self.denormalize_action(norm_action)
        return action

    def normalize_state(self, state):
        th = np.arctan2(state[1], state[0]) / np.pi
        th_dot = state[2] / 8.0
        return np.array([th, th_dot])

    def game_dimension(self):
        state_dim = 2
        action_dim = 1
        return state_dim, action_dim


def run_my_actor_critic():
    logging.basicConfig(level=logging.DEBUG)
    env = gym.make('Pendulum-v0')
    normalizer = GameNormalizer()
    state_dim, action_dim = normalizer.game_dimension()
    controller = ActorCriticAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        discount_rate=0.99,
        hidden_dims=[8, 8],
        learning_rate=1e-3,
        replay_memory_sz=5000,
        batch_sz=32)
    controller.load('./models/', 'pendulum')
    #
    costs = list()
    while True:
        if len(costs) == 5000:
            print(np.median(costs), np.max(costs))
            controller.save('./models/', 'pendulum')
            costs = list()
        if len(costs) == 0:
            env.reset()
            controller.reset()
            action = env.action_space.sample()
        env.render()
        state, cost, _, _ = env.step(action)
        reward = np.exp(cost)
        controller.train(
            normalizer.normalize_state(state),
            normalizer.normalize_action(action),
            reward)
        action = normalizer.denormalize_action(
            controller.choose_action(
                normalizer.normalize_state(state)))
        costs.append(cost)
    env.close()


if __name__ == "__main__":
    run_my_actor_critic()