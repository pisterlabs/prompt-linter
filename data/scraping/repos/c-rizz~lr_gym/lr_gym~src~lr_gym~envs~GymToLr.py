#!/usr/bin/env python3
"""
Base-class for cresting GAzebo-based gym environments.

The provided class must be extended to define a specific environment
"""

from lr_gym.envs.BaseEnv import BaseEnv

import gym
import numpy as np
from typing import Tuple
from typing import Dict
from typing import Any
from typing import Sequence
import time
import os, psutil



class GymToLr(BaseEnv):
    """This is a base-class for implementing lr_gym environments.

    It defines more general methods to be implemented than the original gym.Env class.

    You can extend this class with a sub-class to implement specific environments.
    """
    #TODO: This should be an abstract class, defined via python's ABC

    action_space = None
    observation_space = None
    metadata = None # e.g. {'render.modes': ['rgb_array']}

    def __init__(self, openaiGym_env : gym.Env, stepSimDuration_sec : float = 1, maxStepsPerEpisode = None):
        """Short summary.

        Parameters
        ----------
        maxStepsPerEpisode : int
            maximum number of frames per episode. The step() function will return
            done=True after being called this number of times

        """

        if maxStepsPerEpisode is None:
            if openaiGym_env.spec is not None:
                maxStepsPerEpisode = openaiGym_env.spec.max_episode_steps
            if maxStepsPerEpisode is None and hasattr(openaiGym_env,"_max_episode_steps"):
                maxStepsPerEpisode = openaiGym_env._max_episode_steps
            if maxStepsPerEpisode is None:
                raise RuntimeError("Cannot determine maxStepsPerEpisode from openaiGym_env env, you need to specify it manually")

        super().__init__(   maxStepsPerEpisode = maxStepsPerEpisode,
                            startSimulation = True,
                            simulationBackend = "OpenAiGym",
                            verbose = False,
                            quiet = False)

        self._openaiGym_env = openaiGym_env

        self._actionToDo = None # This will be set by submitAction an then used in step()
        self._lastObservation = None
        self._previousObservation = None #Observation before the last
        self._lastDone = None
        self._lastInfo = None
        self._lastReward = None
        self._stepCount = 0
        self._stepSimDuration_sec = stepSimDuration_sec

        self.action_space = self._openaiGym_env.action_space
        self.observation_space = self._openaiGym_env.observation_space


    def submitAction(self, action) -> None:
        super().submitAction(action)
        self._actionToDo = action


    def checkEpisodeEnded(self, previousState, state) -> bool:
        if self._lastDone is not None:
            ended = self._lastDone
        else:
            ended = False
        ended = ended or super().checkEpisodeEnded(previousState, state)
        return ended


    def computeReward(self, previousState, state, action) -> float:
        if not (state is self._lastObservation and action is self._actionToDo and previousState is self._previousObservation):
            raise RuntimeError("GymToLr.computeReward is only valid if used for the last executed step. And it looks like you tried using it for something else.")
        return self._lastReward

    def getObservation(self, state) -> np.ndarray:
        return state

    def getState(self) -> Sequence:
        return self._lastObservation


    def initializeEpisode(self) -> None:
        pass


    def performStep(self) -> None:
        super().performStep()
        self._previousObservation = self._lastObservation
        self._stepCount += 1
        # time.sleep(1)
        # print(f"Step {self._stepCount}, memory usage = {psutil.Process(os.getpid()).memory_info().rss/1024} KB")
        self._lastObservation, self._lastReward, self._lastDone, self._lastInfo = self._openaiGym_env.step(self._actionToDo)


    def performReset(self) -> None:
        super().performReset()
        self._previousObservation = None
        self._stepCount = 0
        self._lastObservation = self._openaiGym_env.reset()



    def getUiRendering(self) -> Tuple[np.ndarray, float]:
        return self._openaiGym_env.render(mode="rgb_array"), self.getSimTimeFromEpStart()

    def getInfo(self,state=None) -> Dict[Any,Any]:
        """To be implemented in subclass.

        This method is called by the step method. The values returned by it will be appended in the info variable returned bby step
        """
        return self._lastInfo

    def getMaxStepsPerEpisode(self):
        """Get the maximum number of frames of one episode, as set by the constructor."""
        return self._maxStepsPerEpisode

    def setGoalInState(self, state, goal):
        """To be implemented in subclass.

        Update the provided state with the provided goal. Useful for goal-oriented environments, especially when using HER.
        It's used by ToGoalEnvWrapper.
        """
        raise NotImplementedError()

    def getGoalFromState(self, state):
        """To be implemented in subclass.

        Get the goal for the provided state. Useful for goal-oriented environments, especially when using HER.
        """
        raise NotImplementedError()

    def getAchievedGoalFromState(self, state):
        """To be implemented in subclass.

        Get the currently achieved goal from the provided state. Useful for goal-oriented environments, especially when using HER.
        """
        raise NotImplementedError()

    def getPureObservationFromState(self, state):
        """To be implemented in subclass.

        Get the pure observation from the provided state. Pure observation means the observation without goal and achieved goal.
        Useful for goal-oriented environments, especially when using HER.
        """
        raise NotImplementedError()

    def buildSimulation(self, backend : str = "gazebo"):
        """To be implemented in subclass.

        Build a simulation for the environment.
        """
        pass

    def _destroySimulation(self):
        """To be implemented in subclass.

        Destroy a simulation built by buildSimulation.
        """
        pass

    def getSimTimeFromEpStart(self):
        """Get the elapsed time since the episode start."""
        return self._stepCount * self._stepSimDuration_sec

    def close(self):
        self._destroySimulation()

    def seed(self, seed=None):
        if seed is not None:
            self._envSeed = seed
        self._openaiGym_env.seed(seed)
        return [self._envSeed]
