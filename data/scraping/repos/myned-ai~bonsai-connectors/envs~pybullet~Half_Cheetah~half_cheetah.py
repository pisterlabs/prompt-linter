import logging
from typing import Any, Dict


import numpy as np
from gym_connectors import BonsaiConnector, PyBulletSimulator

log = logging.getLogger("half_cheetah")


class HalfCheetah(PyBulletSimulator):
    """ Implements the methods specific to Half Cheetah environment
    """

    # Environment name, from openai-gym
    environment_name = 'HalfCheetahPyBulletEnv-v0'

    def __init__(self, iteration_limit=200, skip_frame=1):
        """ Initializes the Half cheetah environment
        """

        self.bonsai_state = {"obs": [0.0]}
        self.prev_potential: float = None

        super().__init__(iteration_limit, skip_frame)

    def gym_to_state(self, state) -> Dict[str, Any]:
        """ Converts openai environment state to Bonsai state, as defined in inkling
        """

        joint_speeds = self._env.unwrapped.robot.joint_speeds

        joints_at_limit = float(self._env.unwrapped.robot.joints_at_limit)

        potential = float(self._env.unwrapped.potential)
        if self.prev_potential is None:
            self.prev_potential = potential

        progress = potential - self.prev_potential

        self.bonsai_state = {"obs": state.tolist(),
                             "joint_speeds": joint_speeds.tolist(),
                             "joints_at_limit": joints_at_limit,
                             "progress": progress}

        self.prev_potential = potential

        return self.bonsai_state

    def action_to_gym(self, action: Dict[str, Any]):
        """ Converts Bonsai action type into openai environment action.
        """
        j1 = action['j1']
        j2 = action['j2']
        j3 = action['j3']
        j4 = action['j4']
        j5 = action['j5']
        j6 = action['j6']

        # Half Cheetah environment expects an array of actions
        return [j1, j2, j3, j4, j5, j6]

    def get_state(self) -> Dict[str, Any]:
        """ Returns the current state of the environment
        """
        log.debug('get_state: {}'.format(self.bonsai_state))
        return self.bonsai_state

    def initialize_camera(self, distance, yaw, pitch, x=0, y=0, z=0):
        """Initializes the position of Camera
        """
        lookat = [x, y, z]

        self._env.unwrapped._p.resetDebugVisualizerCamera(
            distance, yaw, pitch, lookat)


if __name__ == "__main__":
    """ Creates a Pendulum environment, passes it to the BonsaiConnector 
        that connects to the Bonsai service that can use it as a simulator  
    """
    logging.basicConfig()
    log = logging.getLogger("half_cheetah")
    log.setLevel(level='INFO')

    # if more information is needed, uncomment this
    # gymlog = logging.getLogger("GymSimulator")
    # gymlog.setLevel(level='DEBUG')

    half_cheetah = HalfCheetah()
    connector = BonsaiConnector(half_cheetah)

    while connector.run():
        continue
