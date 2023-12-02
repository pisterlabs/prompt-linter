import numpy as np
import gym
from gym.spaces import Box, Discrete
from abc import abstractmethod

gym.logger.set_level(40)

class BaseEnv(gym.Env):
    def __init__(self,
                 cost_function,
                 model,
                 dim_action,
                 simulation_horizon,
                 discrete=True,
                 seed=np.random.randint(1e6)
                 ):
        """
        Base class for epidemic-based environments.

        Parameters
        ----------
        cost_function: BaseCostFunction
            A cost function.
        model: BaseModel
            An epidemiological model.
        dim_action: int
            Dimension of the action space.
        simulation_horizon: int
            Simulation horizon in days.
        discrete: bool
            Whether the environment uses discrete or continuous actions.

        """

        # Define model
        self.model = model
        self.simulation_horizon = simulation_horizon
        self.seed(seed)

        # Define reward function
        self.cost_function = cost_function

        # reset the env
        self.env_state, self.previous_env_state = None, None
        self.model_state = None
        self.reset()

        self.dim_state = self.env_state.size
        self.observation_space = Box(-np.inf * np.ones([self.dim_state]), np.inf * np.ones([self.dim_state]))
        self.dim_action = dim_action
        if discrete:
            self.action_space = Discrete(dim_action)
        else:
            self.action_space = Box(- np.float32(np.ones([dim_action])), np.float32(np.ones([dim_action])))

    @abstractmethod
    def reset(self):
        """
        Reset the environment and the tracking of data.

        Returns
        -------
        nd.array
           The initial environment state.

        """
        pass

    @abstractmethod
    def _update_env_state(self):
        """
        Update the environment state.

        """
        pass

    @abstractmethod
    def _normalize_env_state(self, env_state):
        pass

    def run_model(self):
        """
        Run model for one step.

        Returns
        -------
        nd.array
            New environment state.
        """
        return self.model.run_n_steps(self.model_state, 1)

    @abstractmethod
    def step(self, action):
        """
        Traditional step function from OpenAI Gym envs. Uses the action to update the environment.

        Parameters
        ----------
        action: int if discrete, nd.array if continuous.

        Returns
        -------
        state: nd.array
            New environment state.
        cost_aggregated: float
            Aggregated measure of the cost.
        done: bool
            Whether the episode is terminated.
        info: dict
            Further infos. In our case, the costs, icu capacity of the region and whether constraints are violated.

        """
        pass
