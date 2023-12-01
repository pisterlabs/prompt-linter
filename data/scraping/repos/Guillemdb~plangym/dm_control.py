import sys
import traceback

from gym.spaces import Box
import numpy as np

from plangym.core import GymEnvironment
from plangym.parallel import BatchEnv, ExternalProcess

try:
    from gym.envs.classic_control import rendering

    novideo_mode = False
except Exception:
    novideo_mode = True


class DMControlEnv(GymEnvironment):
    """
    Wrap the dm_control library so it can work for planning problems.

    It allows parallel and vectorized execution of the environments.

    Args:
        name: Provide the task to be solved as `domain_name-task_name`. For
            example 'cartpole-balance'.
        visualize_reward: match dm_control interface. It modifies the color
            of the robot depending on its current reward.
        dt: Set a deterministic frameskip to apply the same
            action N times.
        custom_death: Class for setting custom boundary conditions to help
            exploration.

    """

    def __init__(
        self,
        name: str = "cartpole-balance",
        visualize_reward: bool = True,
        dt: int = 1,
        custom_death: "CustomDeath" = None,
        *args,
        **kwargs,
    ):

        from dm_control import suite

        domain_name, task_name = name.split("-")
        super(DMControlEnv, self).__init__(name=name, dt=dt, *args, **kwargs)
        self._render_i = 0
        self._env = suite.load(
            domain_name=domain_name, task_name=task_name, visualize_reward=visualize_reward
        )
        self._name = name
        self.viewer = []
        self._last_time_step = None

        self._viewer = None if novideo_mode else rendering.SimpleImageViewer()

        self._custom_death = custom_death
        shape = self.reset(return_state=False).shape
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)

        self.action_space = Box(
            low=self.action_spec().minimum, high=self.action_spec().maximum, dtype=np.float32
        )

        self.reset()

    def __getattr__(self, item):
        return getattr(self._env, item)

    def action_spec(self):
        return self.env.action_spec()

    def action_space(self):
        return self.env.action_spec()

    @property
    def physics(self):
        return self.env.physics

    @property
    def env(self):
        """Access to the environment."""
        return self._env

    def set_seed(self, seed):
        np.random.seed(seed)
        self.env.seed(seed)

    def render(self, mode="human"):
        """
        It stores all the RGB images rendered to be shown when the `show_game`
        function is called.

        Args:
            mode: `rgb_array` return an RGB image stored in a numpy array. `human`
             stores the rendered image in a viewer to be shown when `show_game`
             is called.

        Returns:
            numpy.ndarray when mode == `rgb_array`. True when mode == `human`
        """
        img = self.env.physics.render(camera_id=0)
        if mode == "rgb_array":
            return img
        elif mode == "human":
            self.viewer.append(img)
        return True

    def show_game(self, sleep: float = 0.05):
        import time

        for img in self.viewer:
            self._viewer.imshow(img)
            time.sleep(sleep)

    def reset(self, return_state: bool = None) -> [np.ndarray, tuple]:
        """
        Resets the environment and returns the first observation, or the first
        (state, obs) tuple.

        Args:
            return_state: If true return a also the initial state of the env.

        Returns:
            Observation of the environment if `return_state` is False. Otherwise
            return (state, obs) after reset.
        """
        return_state = self.states_on_reset if return_state is None else return_state
        time_step = self._env.reset()
        observed = self._time_step_to_obs(time_step)
        self._render_i = 0
        if not return_state:
            return observed
        else:
            return self.get_state(), observed

    def set_state(self, state: tuple):
        """
        Sets the state of the simulator to the target State.
        I will be super grateful if someone shows me how to do this using Open Source code.

        Args:
            state: numpy.ndarray containing the information about the state to be set.

        Returns:
            None
        """
        with self.env.physics.reset_context():
            # mj_reset () is  called  upon  entering  the  context.
            self.env.physics.data.qpos[:] = state[0]  # Set  position ,
            self.env.physics.data.qvel[:] = state[1]  # velocity
            self.env.physics.data.ctrl[:] = state[2]  # and  control.

    def get_state(self) -> tuple:
        """
        Returns a tuple containing the three arrays that characterize the state
         of the system. Each tuple contains the position of the robot, its velocity
         and the control variables currently being applied.

        Returns:
            Tuple of numpy arrays containing all the information needed to describe
            the current state of the simulation.
        """
        state = (
            np.array(self.env.physics.data.qpos),
            np.array(self.env.physics.data.qvel),
            np.array(self.env.physics.data.ctrl),
        )
        return state

    def step(self, action: np.ndarray, state: np.ndarray = None, dt: int = None) -> tuple:
        """
        Step the environment applying a given action from an arbitrary state. If
        is not provided the signature matches the one from OpenAI gym. It allows
        to apply arbitrary boundary conditions to define custom end states in case
        the env was initialized with a "CustomDeath' object.

        Args:
            action: Array containing the action to be applied.
            state: State to be set before stepping the environment.
            dt: Consecutive number of times to apply the given action.

        Returns:
            if states is None returns (observs, rewards, ends, infos) else (new_states,
            observs, rewards, ends, infos)
        """
        dt = dt if dt is not None else self.dt

        custom_death = False
        end = False
        cum_reward = 0
        if state is not None:
            self.set_state(state)
        for _ in range(int(dt)):
            time_step = self.env.step(action)
            end = end or time_step.last()
            cum_reward += time_step.reward
            # The death condition is a super efficient way to discard huge chunks of the
            # state space at discretion of the programmer.
            if self._custom_death is not None:
                custom_death = custom_death or self._custom_death.calculate(
                    self, time_step, self._last_time_step
                )
            self._last_time_step = time_step
            if end:
                break
        observed = self._time_step_to_obs(time_step)
        # This is written as a hack because using custom deaths should be a hack.
        if self._custom_death is not None:
            end = end or custom_death

        if state is not None:
            new_state = self.get_state()
            return new_state, observed, cum_reward, end, {"lives": 0, "dt": dt}
        return observed, cum_reward, end, {"lives": 0, "dt": dt}

    def step_batch(self, actions, states=None, dt: [int, np.ndarray] = None) -> tuple:
        """
        Vectorized version of the `step` method. It allows to step a vector of
        states and actions. The signature and behaviour is the same as `step`, but taking
        a list of states, actions and dts as input.

        Args:
            actions: Iterable containing the different actions to be applied.
            states: Iterable containing the different states to be set.
            dt: int or array containing the frameskips that will be applied.

        Returns:
            if states is None
                (observs, rewards, ends, infos)
            else
                (new_states, observs, rewards, ends, infos)
        """
        dt = dt if dt is not None else self.dt
        dt = dt if isinstance(dt, np.ndarray) else np.ones(len(states)) * dt
        data = [self.step(action, state, dt=dt) for action, state, dt in zip(actions, states, dt)]
        new_states, observs, rewards, terminals, lives = [], [], [], [], []
        for d in data:
            if states is None:
                obs, _reward, end, info = d
            else:
                new_state, obs, _reward, end, info = d
                new_states.append(new_state)
            observs.append(obs)
            rewards.append(_reward)
            terminals.append(end)
            lives.append(info)
        if states is None:
            return observs, rewards, terminals, lives
        else:
            return new_states, observs, rewards, terminals, lives

    @staticmethod
    def _time_step_to_obs(time_step) -> np.ndarray:
        # Concat observations in a single array, so it is easier to calculate distances
        obs_array = np.hstack(
            [np.array([time_step.observation[x]]).flatten() for x in time_step.observation]
        )
        return obs_array


class ExternalDMControl(ExternalProcess):
    """I cannot find a way to pass a function that creates a DMControl env, so I have to create
      it manually inside the thread.
      Step environment in a separate process for lock free paralellism.
      The environment will be created in the external process.
      Args:
         name: Name of the Environment.
         wrappers: Wrappers to be applied to the Environment.
         dt: Number of consecutive times that action will be applied.
         *args: Additional args to be passed to the environment.
         **kwargs: Additional kwargs to be passed to the environment.

      Attributes:
          observation_space: The cached observation space of the environment.
          action_space: The cached action space of the environment.
     """

    def __init__(self, name, wrappers=None, dt: int = 1, *args, **kwargs):

        self.name = name
        super(ExternalDMControl, self).__init__(constructor=(name, wrappers, dt, args, kwargs))

    def _worker(self, data, conn):
        """The process waits for actions and sends back environment results.
        Args:
          data: tuple containing the necessary parameters.
          conn: Connection for communication to the main process.
        Raises:
          KeyError: When receiving a message of unknown type.
        """
        try:
            name, wrappers, dt, args, kwargs = data

            env = DMControlEnv(name, dt=dt, *args, **kwargs)
            # dom_name, task_name = name.split("-")
            # custom_death = CustomDeath(domain_name=dom_name,
            #                             task_name=task_name)
            env.reset()
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    assert payload is None
                    break
                raise KeyError("Received message of unknown type {}".format(message))
        except Exception:  # pylint: disable=broad-except
            # import tensorflow as tf
            # TODO: use logging to report exceptions.
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print("Error in environment process: {}".format(stacktrace))
            conn.send((self._EXCEPTION, stacktrace))
            conn.close()


class ParallelDMControl(GymEnvironment):
    """Wrap a dm_control environment to be stepped in parallel. It contains a
    :class: DMControlEnv that performs non-vectorized operations, and a :class: BatchEnv
    that performs the `step_batch` method asynchronously.

    Args:
            name: Name of the Environment. following the same conventions as
                :class: DMControlEnv.
            dt: Number of consecutive times that action will be applied.
            n_workers: Number of processes that will be used.
            blocking: if False step the environments asynchronously.
            *args: args of the environment that will be parallelized.
            **kwargs: kwargs of the environment that will be parallelized.
    """

    def __init__(
        self, name: str, dt: int = 1, n_workers: int = 8, blocking: bool = True, *args, **kwargs
    ):

        super(ParallelDMControl, self).__init__(name=name)

        envs = [ExternalDMControl(name=name, dt=dt, *args, **kwargs) for _ in range(n_workers)]
        self._batch_env = BatchEnv(envs, blocking)
        self._env = DMControlEnv(name, dt=dt, *args, **kwargs)

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def __getattr__(self, item):
        return getattr(self._env, item)

    def step_batch(
        self, actions: np.ndarray, states: np.ndarray = None, dt: [np.ndarray, int] = None,
    ):
        """
        Vectorized version of the `step` method. It allows to step a vector of
        states and actions. The signature and behaviour is the same as `step`, but taking
        a list of states, actions and dts as input.

        Args:
           actions: Iterable containing the different actions to be applied.
           states: Iterable containing the different states to be set.
           dt: int or array containing the frameskips that will be applied.

        Returns:
          if states is None returns (observs, rewards, ends, infos) else (new_states,
          observs, rewards, ends, infos)
        """
        return self._batch_env.step_batch(actions=actions, states=states, dt=dt)

    def step(self, action: np.ndarray, state: np.ndarray = None, dt: int = None):
        """
        Step the environment applying a given action from an arbitrary state. If
        is not provided the signature matches the one from OpenAI gym. It allows
        to apply arbitrary boundary conditions to define custom end states in case
        the env was initialized with a "CustomDeath' object.

        Args:
            action: Array containing the action to be applied.
            state: State to be set before stepping the environment.
            dt: Consecutive number of times to apply the given action.

        Returns:
            if states is None returns (observs, rewards, ends, infos) else (new_states,
            observs, rewards, ends, infos)
        """
        return self._env.step(action=action, state=state, dt=dt)

    def reset(self, return_state: bool = True, blocking: bool = True):
        """
        Resets the environment and returns the first observation, or the first
        (state, obs) tuple, and synchronized the states of all the workers to
        match the state of the internal :class: DMControlEnv.
        Args:
            return_state: If true return a also the initial state of the env.
            blocking: If False, reset the environments asynchronously.

        Returns:
            Observation of the environment if `return_state` is False. Otherwise
            return (state, obs) after reset.
        """
        state, obs = self._env.reset(return_state=True)
        self.sync_states()
        return state, obs if return_state else obs

    def get_state(self):
        """
        Returns a tuple containing the three arrays that characterize the state
         of the system. Each tuple contains the position of the robot, its velocity
         and the control variables currently being applied.

        Returns:
            Tuple of numpy arrays containing all the information needed to describe
            the current state of the simulation.
        """
        return self._env.get_state()

    def set_state(self, state):
        """Sets the state of the underlying :class: DMControlEnv and the states of all the
        workers used by the internal :class: BatchEnv."""
        self._env.set_state(state)
        self.sync_states()

    def sync_states(self):
        """Set all the states of the different workers of the internal :class: BatchEnv to the
        same state as the internal :class: DMControlEnv used to apply the
        non-vectorized steps."""
        self._batch_env.sync_states(self.get_state())


class CustomDeath:

    """Class for taking into account arbitrary boundary conditions."""

    def __init__(self, domain_name="cartpole", task_name="balance"):
        self._domain_name = domain_name
        self._task_name = task_name

    @property
    def task_name(self):
        return self._task_name

    @property
    def domain_name(self):
        return self._domain_name

    def calculate(self, env: DMControlEnv, time_step, last_time_step):

        if self._domain_name == "cartpole" and self.task_name == "balance":
            return self._cartpole_balance_death(env=env, time_step=time_step)
        elif self._domain_name == "hopper":
            return self._hopper_death(env=env, time_step=time_step, last_time_step=last_time_step)
        elif self._domain_name == "walker":
            return self._walker_death(env=env, time_step=time_step, last_time_step=last_time_step)
        else:
            return self._default_death(time_step, last_time_step)

    @staticmethod
    def _default_death(time_step, last_time_step) -> bool:
        last_rew = last_time_step.reward if last_time_step is not None else 0
        return time_step.reward <= 0 and last_rew > 0

    @staticmethod
    def _cartpole_balance_death(env, time_step) -> bool:
        """If the reward is less than 0.7 consider a state dead. This threshold is because rewards
        lesser than 0.7 involve positions where the cartpole is not balanced.
        """
        return time_step.reward < 0.75 or abs(env.physics.cart_position()) > 0.5

    @staticmethod
    def _hopper_death(env, time_step, last_time_step) -> bool:
        # min_torso_height = 0.1
        # max_reward_drop = 0.3

        # torso_touches_ground = env.physics.height() < min_torso_height
        # reward_change = time_step.reward - (last_time_step.reward if
        # last_time_step is not None else 0)
        # reward_drops = reward_change < -max_reward_drop * env.dt
        return False

    @staticmethod
    def _walker_death(env, time_step, last_time_step) -> bool:
        min_torso_height = 0.1
        max_reward_drop_pct = 0.5
        # max_tilt = 0
        min_reward = 0.1

        torso_touches_ground = env.physics.torso_height() < min_torso_height
        last_reward = last_time_step.reward if last_time_step is not None else 0.00001
        reward_change = time_step.reward / last_reward
        reward_drops = reward_change < max_reward_drop_pct
        torso_very_tilted = False  # abs(env.physics.torso_upright())
        # < max_tilt and reward_change < 0
        # torso_very_tilted = torso_very_tilted if not env.state.dead else False

        crappy_reward = time_step.reward < min_reward  # if not env.state.dead else False

        return reward_drops or torso_touches_ground or torso_very_tilted or crappy_reward
