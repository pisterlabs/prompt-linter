from gym.envs.classic_control import PendulumEnv
import jax.numpy as jp

from research.estop import ddpg
from research.statistax import Deterministic, Uniform

def cost(s, u):
  return (s[0] - jp.pi)**2 + 0.1 * (s[1]**2) + 0.001 * (u[0]**2)

def pendulum_environment(mass: float,
                         length: float,
                         gravity: float,
                         friction: float,
                         max_speed: float,
                         dt: float,
                         reward_adjustment: float = 0) -> ddpg.Env:
  """A pendulum swing up environment. This requires a swing-up when `gravity` is
  greater than `max_torque / (mass * length)` and can overpower gravity
  otherwise."""
  def step(state, action):
    """Take a single step in the discretized pendulum dynamics.

    Args:
      state (ndarray): An ndarray with the current theta, and d theta/dt. Note
        that theta ranges from 0 to 2 pi, with 0 and 2 pi denoting the bottom of
        the pendulum swing and pi denoting the top.
      action (ndarray): The force to be applied. Positive force going
        counterclockwise and negative force going clockwise.

    Returns:
      A Distribution over next states.
    """
    assert state.shape == (4, )
    assert action.shape == (1, )

    theta = state[0]
    theta_dot = state[1]
    u = action[0]

    theta_dotdot = (-gravity / length * jp.sin(theta) - friction * theta_dot + u /
                    (mass * length**2))

    # Slightly different from OpenAI gym since we clip before adding to theta.
    new_theta_dot = jp.clip(theta_dot + dt * theta_dotdot, -max_speed, max_speed)
    new_theta = (theta + dt * new_theta_dot) % (2 * jp.pi)
    return Deterministic(
        jp.array([
            new_theta,
            new_theta_dot,
            jp.cos(new_theta),
            jp.sin(new_theta),
        ]))

  # In the initial state we force the x/y coordinates to be zero which is a lie,
  # but oh well.
  return ddpg.Env(
      initial_distribution=Uniform(
          jp.array([jp.pi - 0.1, -1, 1, 0]),
          jp.array([jp.pi + 0.1, 1, 1, 0]),
      ),
      # initial_distribution=Uniform(
      #     jp.array([0, -1, 0, 0]),
      #     jp.array([2 * jp.pi, 1, 0, 0]),
      # ),
      step=step,
      reward=lambda s1, a, _: 1.0 - cost(s1, a) / reward_adjustment,
  )

def viz_pendulum_rollout(states, actions):
  assert states.shape[0] == actions.shape[0]

  eps = jp.finfo(float).eps

  gymenv = PendulumEnv()
  gymenv.reset()

  for t in range(states.shape[0]):
    gymenv.state = states[t] + jp.pi
    # array(0.0) is False-y which causes problems.
    gymenv.last_u = actions[t] + eps
    gymenv.render()

  gymenv.close()
