from reinforch.agents import PolicyGradientAgent
from reinforch.core.logger import Log, INFO
from reinforch.core.memorys import PGMemory
from reinforch.environments import OpenAIGym
from reinforch.execution import Runner
from examples import default_config_path, default_save_folder

logger = Log(__name__, level=INFO)

if __name__ == '__main__':
    def reward_shape(state=None, reward=None, terminal=None, env=None):
        x, x_dot, theta, theta_dot = state
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        return reward


    gym_id = 'CartPole-v0'

    env = OpenAIGym(gym_id, reward_shape=reward_shape)
    env.seed(7)
    n_s = env.n_s
    n_a = env.n_a
    memory = PGMemory()

    agent = PolicyGradientAgent(n_s=n_s,
                                n_a=n_a,
                                memory=memory,
                                config=default_config_path('pg', gym_id))
    with Runner(agent=agent,
                environment=env,
                save_dest_folder=default_save_folder('pg', gym_id),
                verbose=False) as runner:
        runner.train(total_episode=500,
                     save_final_model=True,
                     visualize=False)

        logger.info('The agent has completed its training...')

        runner.test(total_episode=10, visualize=True)
