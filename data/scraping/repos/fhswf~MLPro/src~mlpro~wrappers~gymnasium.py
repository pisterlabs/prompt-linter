## -------------------------------------------------------------------------------------------------
## -- Project : MLPro - A Synoptic Framework for Standardized Machine Learning Tasks
## -- Package : mlpro.wrappers
## -- Module  : gymnasium.py
## -------------------------------------------------------------------------------------------------
## -- History :
## -- yyyy-mm-dd  Ver.      Auth.    Description
## -- 2023-04-09  0.0.0     MRD      Copied from openai_gym wrapper
## -- 2023-04-10  1.0.0     MRD      First Release 
## -- 2023-04-19  1.0.1     MRD      Refactor reset function
## -- 2023-08-21  1.0.2     MRD      Saving the seed in variable self._p_seed
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.2 (2023-08-21)

This module provides wrapper classes for Gym environments from Farama-Foundation Gymnasium.

See also: https://github.com/Farama-Foundation/Gymnasium

"""

import gymnasium as gym
from mlpro.wrappers.models import Wrapper
from mlpro.rl import *




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrEnvGYM2MLPro(Wrapper, Environment):
    """
    This class is a ready to use wrapper class for Gym environments. 
    Objects of this type can be treated as an environment object. Encapsulated gym environment 
    must be compatible to class gym.Env.

    Parameters
    ----------
    p_gym_env : Env   
        Gym environment object
    p_state_space : MSpace  
        Optional external state space object that meets the state space of the Gym environment
    p_action_space : MSpace 
        Optional external action space object that meets the action space of the Gym environment
    p_visualize : bool
        Boolean switch for env/agent visualisation. Default = True.
    p_logging
        Log level (see constants of class Log). Default = Log.C_LOG_ALL.
    """

    C_TYPE              = 'Wrapper Gym2MLPro'
    C_WRAPPED_PACKAGE   = 'gymnasium'
    C_MINIMUM_VERSION   = '0.28.1'
    C_PLOT_ACTIVE: bool = True

## -------------------------------------------------------------------------------------------------
    def __init__(self,
                 p_gym_env,  
                 p_state_space: MSpace = None,  
                 p_action_space: MSpace = None,  
                 p_seed=None,
                 p_visualize:bool=True,
                 p_logging=Log.C_LOG_ALL):

        self._gym_env = p_gym_env
        self._p_seed = p_seed
            
        self._gym_env_id = self._gym_env.env.spec.id
        self.C_NAME      = '(' + self._gym_env_id + ')'

        Environment.__init__(self, p_mode=Environment.C_MODE_SIM, p_latency=None, p_visualize=p_visualize, p_logging=p_logging)
        Wrapper.__init__(self, p_logging=p_logging)

        if p_state_space is not None:
            self._state_space = p_state_space
        else:
            self._state_space = self.recognize_space(self._gym_env.observation_space)

        if p_action_space is not None:
            self._action_space = p_action_space
        else:
            self._action_space = self.recognize_space(self._gym_env.action_space)
        
        self.log(self.C_LOG_TYPE_I, 'Gym Environment has been sucessfully wrapped as MLPro Environment.')


## -------------------------------------------------------------------------------------------------
    def __del__(self):
        """
        To shut down or close an environment.

        """
        try:
            self._gym_env.close()
            self.log(self.C_LOG_TYPE_I, 'Closed')
        except:
            pass


## -------------------------------------------------------------------------------------------------
    def _reduce_state(self, p_state:dict, p_path:str, p_os_sep:str, p_filename_stub:str):
        """
        The embedded Gym env itself can't be pickled due to it's dependencies on Pygame. That's why
        the current env instance needs to be removed before pickling the object. 

        See also: https://stackoverflow.com/questions/52336196/how-to-save-object-using-pygame-surfaces-to-file-using-pickle
        """

        p_state['_gym_env'] = None


## -------------------------------------------------------------------------------------------------
    def _complete_state(self, p_path:str, p_os_sep:str, p_filename_stub:str):
        self._gym_env = gym.make(self._gym_env_id)


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def recognize_space(p_gym_space) -> ESpace:
        """
        Detecting a gym space and transform it to MLPro space. Hence, the transformed space can be
        directly compatible in MLPro.

        Parameters
        ----------
        p_gym_space : container spaces (:class:`Tuple` and :class:`Dict`)
            Spaces are crucially used in Gym to define the format of valid actions and observations.

        Returns
        -------
        space : ESpace
            MLPro compatible space.

        """
        space = ESpace()

        if isinstance(p_gym_space, gym.spaces.Discrete):
            space.add_dim(
                Dimension(p_name_short='0', p_base_set=Dimension.C_BASE_SET_Z, p_boundaries=[0, int(p_gym_space.n-1)]))
        elif isinstance(p_gym_space, gym.spaces.Box):
            shape_dim = len(p_gym_space.shape)
            for i in range(shape_dim):
                for d in range(p_gym_space.shape[i]):
                    space.add_dim(Dimension(p_name_short=str(d), p_base_set=Dimension.C_BASE_SET_R,
                                            p_boundaries=[p_gym_space.low[d], p_gym_space.high[d]]))

        return space


## -------------------------------------------------------------------------------------------------
    @staticmethod
    def setup_spaces():
        """
        To setup spaces. To be optionally defined by the users.

        """
        return None, None


## -------------------------------------------------------------------------------------------------
    def _reset(self, p_seed=None):
        """
        Custom method to reset the environment to an initial/defined state. 

        Parameters
        ----------
        p_seed : int
            Seed parameter for an internal random generator. Default = None.

        """

        # 1 Reset Gym environment and determine initial state
        try:
            observation, _ = self._gym_env.reset(seed=p_seed)
        except:
           self._gym_env.seed(p_seed)
           observation = self._gym_env.reset() 
        obs = DataObject(observation)

        # 2 Create state object from Gym observation
        state = State(self._state_space)
        state.set_values(obs.get_data())
        self._set_state(state)


## -------------------------------------------------------------------------------------------------
    def simulate_reaction(self, p_state: State, p_action: Action) -> State:
        """
        Simulates a state transition based on a state and an action. The simulation step itself is
        carried out either by an internal custom implementation in method _simulate_reaction() or
        by an embedded adaptive function.

        Parameters
        ----------
        p_state : State
            Current state.
        p_action : Action
            Action.

        Returns
        -------
        state : State
            Subsequent state after transition

        """

        # 1 Convert action to Gym syntax
        action_sorted = p_action.get_sorted_values()
        dtype = self._gym_env.action_space.dtype

        if (dtype == np.int32) or (dtype == np.int64):
            action_sorted = action_sorted.round(0)

        if action_sorted.size == 1:
            action_gym = action_sorted.astype(self._gym_env.action_space.dtype)[0]
        else:
            action_gym = action_sorted.astype(self._gym_env.action_space.dtype)

        # 2 Process step of Gym environment
        try:
            observation, reward_gym, termination, truncation, info = self._gym_env.step(action_gym)
        except:
            observation, reward_gym, termination, truncation, info = self._gym_env.step(np.atleast_1d(action_gym))
        obs = DataObject(observation)

        # 3 Create state object from Gym observation
        state = State(self._state_space, p_terminal=termination, p_timeout=truncation)
        state.set_values(obs.get_data())

        # 4 Create reward object
        self._last_reward = Reward(Reward.C_TYPE_OVERALL)
        self._last_reward.set_overall_reward(reward_gym)

        # 5 Return next state
        return state


## -------------------------------------------------------------------------------------------------
    def compute_reward(self, p_state_old: State = None, p_state_new: State = None) -> Reward:
        """
        Computes a reward for the state transition, given by two successive states. The reward
        computation itself is carried out either by a custom implementation in method
        _compute_reward() or by an embedded adaptive function.

        Parameters
        ----------
        p_state_old : State
            Optional state before transition. If None the internal previous state of the environment
            is used.
        p_state_new : State
            Optional tate after transition. If None the internal current state of the environment
            is used.

        Returns
        -------
        Reward
            Reward object.

        """
        if (p_state_old is not None) or (p_state_new is not None):
            raise NotImplementedError

        return self._last_reward


## -------------------------------------------------------------------------------------------------
    def compute_success(self, p_state: State) -> bool:
        """
        Assesses the given state whether it is a 'success' state. Assessment is carried out either by
        a custom implementation in method _compute_success() or by an embedded adaptive function.

        Parameters
        ----------
        p_state : State
            State to be assessed.

        Returns
        -------
        bool
            True, if the given state is a 'success' state. False otherwise.

        """

        return self.get_success()


## -------------------------------------------------------------------------------------------------
    def compute_broken(self, p_state: State) -> bool:
        """
        Assesses the given state whether it is a 'broken' state. Assessment is carried out either by
        a custom implementation in method _compute_broken() or by an embedded adaptive function.

        Parameters
        ----------
        p_state : State
            State to be assessed.

        Returns
        -------
        bool
            True, if the given state is a 'broken' state. False otherwise.
            
        """
        
        return self.get_broken()


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure: Figure = None, p_plot_settings: list = ..., p_plot_depth: int = 0, p_detail_level: int = 0, p_step_rate: int = 0, **p_kwargs):
        if self._visualize: self._gym_env.render()


## -------------------------------------------------------------------------------------------------
    def update_plot(self, **p_kwargs):
        """
        Updating the actual plot, deployed by render functionality from Gym.

        """
        if self._visualize: self._gym_env.render()


## -------------------------------------------------------------------------------------------------
    def get_cycle_limit(self):
        """
        To obtain the information regarding the cycle limit from the environment.

        Returns
        -------
        float
            the number of the cycle limit.

        """
        return self._gym_env._max_episode_steps





## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class WrEnvMLPro2GYM(Wrapper, gym.Env):
    """
    This class is a ready to use wrapper class for MLPro to Gym environments. 
    Objects of this type can be treated as an gym.Env object. Encapsulated 
    MLPro environment must be compatible to class Environment.

    Parameters
    ----------
    p_mlpro_env : Environment    
            MLPro's Environment object
    p_state_space : MSpace  
            Optional external state space object that meets the state space of the MLPro environment
    p_action_space : MSpace 
            Optional external action space object that meets the state space of the MLPro environment
    p_render_mde : str
            To allow the user to specify render_mode handled by the environment, for instance,
            'human', 'rgb_array', and 'single_rgb_array'. Default = None.
    p_logging
            Log level (see constants of class Log). Default = Log.C_LOG_ALL.
    """

    C_TYPE              = 'Wrapper MLPro2Gym'
    C_WRAPPED_PACKAGE   = 'gymnasium'
    C_MINIMUM_VERSION   = '0.28.1'
    metadata            = {'render.modes': ['human', 'rgb_array']}

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_mlpro_env:Environment, 
                  p_state_space: MSpace = None, 
                  p_action_space: MSpace = None, 
                  p_render_mode: str = None,
                  p_logging = Log.C_LOG_ALL ):

        Wrapper.__init__(self, p_logging=p_logging)

        self._mlpro_env = p_mlpro_env

        if p_state_space is not None:
            self.observation_space = p_state_space
        else:
            self.observation_space = self.recognize_space(self._mlpro_env.get_state_space())

        if p_action_space is not None:
            self.action_space = p_action_space
        else:
            self.action_space = self.recognize_space(self._mlpro_env.get_action_space())

        if p_render_mode is not None:
            self.render_mode = p_render_mode
        else:
            self.render_mode = None

        self.first_refresh = True
    
        self.log(self.C_LOG_TYPE_I, 'MLPro Environment has been sucessfully wrapped as Gym Environment.')

    
## -------------------------------------------------------------------------------------------------
    @staticmethod
    def recognize_space(p_mlpro_space):
        """
        Detecting a MLPro space and transform it to gym space. Hence, the transformed space can be
        directly compatible in gym.

        Parameters
        ----------
        p_mlpro_space : ESpace
            MLPro compatible space.

        Returns
        -------
        space : container spaces (:class:`Tuple` and :class:`Dict`)
            Spaces are crucially used in Gym to define the format of valid actions and observations.

        """
        space = None
        action_dim = p_mlpro_space.get_num_dim()
        id_dim = p_mlpro_space.get_dim_ids()[0]
        base_set = p_mlpro_space.get_dim(id_dim).get_base_set()
        if len(p_mlpro_space.get_dim(id_dim).get_boundaries()) == 1:
            space = gym.spaces.Discrete(p_mlpro_space.get_dim(id_dim).get_boundaries()[0]+1)
        else:
            lows = []
            highs = []
            for dimension in range(action_dim):
                id_dim = p_mlpro_space.get_dim_ids()[dimension]
                lows.append(p_mlpro_space.get_dim(id_dim).get_boundaries()[0])
                highs.append(p_mlpro_space.get_dim(id_dim).get_boundaries()[1])

            space = gym.spaces.Box(
                low=np.array(lows, dtype=np.float32),
                high=np.array(highs, dtype=np.float32),
                shape=(action_dim,),
                dtype=np.float32
            )

        return space

    
## -------------------------------------------------------------------------------------------------
    def step(self, action):
        """
        To execute one time step within the environment.

        Parameters
        ----------
        action : ActType
            an action provided by the agent.

        Returns
        -------
        obs : object
            This will be an element of the environment's :attr:`observation_space`.
            This may, for instance, be a numpy array containing the positions and velocities of certain objects.
        reward.get_overall_reward() : float
            The amount of reward returned as a result of taking the action.
        terminated : bool
            whether a `terminal state` (as defined under the MDP of the task) is reached.
            In this case further step() calls could return undefined results.
        truncated : bool
            whether a truncation condition outside the scope of the MDP is satisfied.
            Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
            Can be used to end the episode prematurely before a `terminal state` is reached.
        info : dict
            It contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
            This might, for instance, contain: metrics that describe the agent's performance state, variables that are
            hidden from observations, or individual reward terms that are combined to produce the total reward.
            It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
            of returning two booleans, and will be removed in a future version.

        """
        _action = Action()
        _act_set = Set()
        idx = self._mlpro_env._action_space.get_num_dim()

        if isinstance(self.observation_space, gym.spaces.Discrete):
            action = np.array([action])

        for i in range(idx):
            _act_set.add_dim(Dimension('action_' + str(i)))
        _act_elem = Element(_act_set)
        for i in range(idx):
            _ids = _act_elem.get_dim_ids()
            _act_elem.set_value(_ids[i], action[i].item())
        _action.add_elem('0', _act_elem)

        self._mlpro_env.process_action(_action)
        reward = self._mlpro_env.compute_reward()

        obs = None
        if isinstance(self.observation_space, gym.spaces.Box):
            obs = np.array(self._mlpro_env.get_state().get_values(), dtype=np.float32)
        else:
            obs = np.array(self._mlpro_env.get_state().get_values())

        state = self._mlpro_env.get_state()
        terminated = state.get_terminal()
        truncated = state.get_timeout()

        info = {}

        return obs, reward.get_overall_reward(), terminated, truncated, info


    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.
        This is for new gym version.

        Parameters
        ----------
        seed : int, optional
           The seed that is used to initialize the environment's PRNG.
           If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
           a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
           However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
           If you pass an integer, the PRNG will be reset even if it already exists.
           Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
           Please refer to the minimal example above to see this paradigm in action.
           The default is None.
        return_info : bool
            If true, return additional information along with initial observation.
            This info should be analogous to the info returned in :meth:`step`.
            The default is False.
        options : dict, optional
            Additional information to specify how the environment is reset (optional,
            depending on the specific environment). The default is None.

        Returns
        -------
        obs : object
            This will be an element of the environment's :attr:`observation_space`.
            This may, for instance, be a numpy array containing the positions and velocities of certain objects.
        info : dict
            It contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
            This might, for instance, contain: metrics that describe the agent's performance state, variables that are
            hidden from observations, or individual reward terms that are combined to produce the total reward.
            It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
            of returning two booleans, and will be removed in a future version.

        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self._mlpro_env.reset(seed)
        obs = None
        if isinstance(self.observation_space, gym.spaces.Box):
            obs = np.array(self._mlpro_env.get_state().get_values(), dtype=np.float32)
        else:
            obs = np.array(self._mlpro_env.get_state().get_values())
        
        info = {}
        return obs, info

    
## -------------------------------------------------------------------------------------------------
    def render(self, mode='human'):
        """
        Compute the render frames as specified by render_mode attribute during initialization of the environment.

        Parameters
        ----------
        mode : str, optional
            To allow the user to specify render_mode handled by the environment, for instance,
            'human', 'rgb_array', and 'single_rgb_array'. The default is 'human'.

        Returns
        -------
        bool
            Rendering is successful or not.

        """
        try:
            if self.first_refresh:
                self._mlpro_env.init_plot()
                self.first_refresh = False
            else:
                self._mlpro_env.update_plot()
            return True
        except:
            return False

    
## -------------------------------------------------------------------------------------------------
    def close(self):
        """
        Override close in your subclass to perform any necessary cleanup.
        Environments will automatically :meth:`close()` themselves when garbage collected or when the program exits.

        """
        self._mlpro_env.__del__()