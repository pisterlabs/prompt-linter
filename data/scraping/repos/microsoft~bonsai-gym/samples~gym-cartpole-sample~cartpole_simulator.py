import sys
import logging
from microsoft_bonsai_api.simulator.client import BonsaiClientConfig
from bonsai_gym import GymSimulator3

log = logging.getLogger("gym_simulator")
log.setLevel(logging.DEBUG)


class CartPole(GymSimulator3):
    # Environment name, from openai-gym
    environment_name = "CartPole-v0"

    # Simulator name from Inkling
    simulator_name = "CartpoleSimulator"

    # convert openai gym observation to our state type
    def gym_to_state(self, observation):
        state = {
            "position": observation[0],
            "velocity": observation[1],
            "angle": observation[2],
            "rotation": observation[3],
        }
        return state

    # convert our action type into openai gym action
    def action_to_gym(self, action):
        return action["command"]


if __name__ == "__main__":
    # create a brain, openai-gym environment, and simulator
    config = BonsaiClientConfig(argv=sys.argv)
    sim = CartPole(config)
    sim.run_gym()
