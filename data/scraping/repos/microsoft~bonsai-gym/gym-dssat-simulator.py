import sys
import logging
import random

# For manual debugging
gym_lib = '/opt/gym_dssat_pdi/lib/python3.9/site-packages'
if gym_lib not in sys.path:
    sys.path.append(gym_lib)
import gym

from microsoft_bonsai_api.simulator.client import BonsaiClientConfig
from bonsai_gym import GymSimulator3

log = logging.getLogger("gym_simulator")
log.setLevel(logging.DEBUG)


class GymDSSAT(GymSimulator3):
    # Environment name, from openai-gym
    environment_name = "gym_dssat_pdi:GymDssatPdi-v0"

    # Simulator name from Inkling
    simulator_name = "DSSATSimulator"

    def __init__(
        self, config: BonsaiClientConfig, iteration_limit: int = 0, skip_frame: int = 1,
    ) -> None:
        self.mode = 'uninitialized'
        super().__init__(config)
        self.init_mode('all')

    def init_mode(self, mode):
        if self.mode == mode:
            return # already in this mode

        print(f'changing mode from {self.mode} to {mode}')

        # create and reset the gym environment
        env_args = {
            'run_dssat_location': '/opt/dssat_pdi/run_dssat',
            'log_saving_path': '/dssat_pdi.log',
            'mode': mode,
            'seed': random.randint(0, 1000000),
            'random_weather': True,
            'auxiliary_file_paths': ['/opt/gym_dssat_pdi/samples/test_files/GAGR.CLI'],
        }

        self._env = gym.make(self.environment_name, **env_args)

        self.mode = mode
        initial_observation = self._env.reset()

        # store initial gym state
        try:
            # initial reward = 0; initial terminal False
            state = self.gym_to_state(initial_observation)
        except NotImplementedError as e:
            raise e
        self._set_last_state(state, 0, False)
        self.iteration_count = 0

    # convert openai gym observation to our state type
    def gym_to_state(self, observation):
        print(f"observation: mode {self.mode}, {observation}")
        if observation:
            # Add mode variable
            if self.mode == 'fertilization':
                observation['mode'] = 1
            elif self.mode == 'irrigation':
                observation['mode'] = 2
            elif self.mode == 'all':
                observation['mode'] = 3
            else:
                observation['mode'] = 0

            # Add cleach variable
            if 'cleach' in self._env._state:
                observation['cleach'] = self._env._state['cleach']

            if observation.get('dap', 0) == 4377089:
                # Correct strange very high value of dap
                # TODO: Not sure why this happens.
                print("dap is 4377089. Strange that this is the first observation. We'll reassign it to 0.")
                observation['dap'] = 0
        return observation

    # convert our action type into openai gym action
    def action_to_gym(self, action):
        print(f"action: {action}")
        return action

    def gym_episode_start(self, parameters):
        print("super gym_episode_start")

        # If a different mode is requested, use init_mode to reinitialize the environment.
        if 'mode' in parameters:
            if parameters['mode'] == 1:
                self.init_mode('fertilization')
            elif parameters['mode'] == 2:
                self.init_mode('irrigation')
            elif parameters['mode'] == 3:
                self.init_mode('all')

        # Initialize the seed to a specific value (if requested) or a random value.
        if parameters.get('seed', 0) != 0:
            print(f'setting seed to {parameters["seed"]}')
            self._env.seed(parameters['seed'])
        else:
            self._env.seed(random.randint(0, 1000000))

        observation = super().gym_episode_start(parameters)
        self.previous_observation = observation
        return observation

    def gym_simulate(self, gym_action):
        observation, reward, done, info = super().gym_simulate(gym_action)

        # The DSSAT gym environment returns some unusual values that the Bonsai
        # gym simulator doesn't like. We'll fix them here.

        if isinstance(reward, list):
            # In the all mode (both irrigation and fertilization) the reward is
            # a list of two values. We'll add them together.
            reward = sum(reward)

        if done:
            # When done, None values are returned.
            if reward == None:
                reward = 0 # None->0 reward

            if observation == None:
                observation = self.previous_observation # use previous observation

        self.previous_observation = observation
        return observation, reward, done, info

if __name__ == "__main__":
    # create a brain, openai-gym environment, and simulator
    print("main")
    config = BonsaiClientConfig(argv=sys.argv)
    sim = GymDSSAT(config)
    sim.run_gym()
