import time

from slm_lab.agent import Agent
from slm_lab.env import OpenAIEnv
from slm_lab.experiment import analysis
from slm_lab.experiment.monitor import Body, InfoSpace, enable_aeb_space
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util

class Session:
    '''The class which initializes the agent, environment, and runs them.'''

    def __init__(self, spec, info_space):
        self.spec = spec
        self.info_space = info_space
        self.index = self.info_space.get('session')

        # init singleton agent and env
        self.env = OpenAIEnv(self.spec)
        body = Body(self.env, self.spec['agent'])
        self.agent = Agent(self.spec, self.info_space, body=body)

        enable_aeb_space(self)  # to use lab's data analysis framework
        logger.info(f'Initialized session {self.index}')

    def run_episode(self):
        self.env.clock.tick('epi')
        reward, state, done = self.env.reset()
        self.agent.reset(state)
        while not done:
            self.env.clock.tick('t')
            action = self.agent.act(state)
            reward, state, done = self.env.step(action)
            self.agent.update(action, reward, state, done)
        self.agent.body.log_summary()

    def close(self):
        self.agent.close()
        self.env.close()
        logger.info('Session done and closed.')

    def run(self, time_limit):
        t0 = time.time()
        while time.time() < t0 + time_limit:
            self.run_episode()
        self.data = analysis.analyze_session(self)  # session fitness
        self.close()
        return self.data, self.agent

    def update_env(self, env):
        self.env.u_env = env
        self.agent.body.env.u_env = env
