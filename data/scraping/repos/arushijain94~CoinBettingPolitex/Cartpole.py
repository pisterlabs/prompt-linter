#######################################################################
# Copyright (C)                                                       #
# 2021 Johann Huber (huber.joh@hotmail.fr)                            #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
"""
Credits:
    The Cart and Pole environment's code has been taken from openai gym source code and modified to add constraints.
        Link : https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L7
    The tile coding software has been taken from Sutton's website.
        Link : http://www.incompleteideas.net/tiles/tiles3.html
"""

import math
import numpy as np


#############################################################################################
#                                     4. Cart and Pole                                      #
#############################################################################################

class CartPoleEnvironment:
    '''
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

     Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

    Constraints:
        Constraint 1: if Cart Position \in {[-2.4, -2.2], [-1.3, -1.1], [-0.1, 0.1], [1.1, 1.3], [2.2, 2.4]},
        then agent getscost=0
            V(Cost) >=40


    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
    '''

    def __init__(self):

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # Position at which to fail the episode
        self.x_threshold = 2.4
        # Action space
        self._all_actions = [0, 1]  # left, right
        self.A = len(self._all_actions)
        # Number of constraints
        self._num_constraints = 2
        # Threshold of constraint2
        self.b = [79, 80]
        # max number of steps for termination
        self._max_steps = 200
        # Discount factor
        self.gamma = 0.99

    def is_state_valid(self, state):
        x, _, theta, _ = state
        # Velocities aren't bounded, therefore cannot be checked.
        is_state_invalid = bool(
            x < -4.8
            or x > 4.8
            or theta < -0.418
            or theta > 0.418
        )
        return not is_state_invalid

    def is_constraint1_violated(self, state):
        """Constraint 1"""
        x, _, _, _ = state
        is_constraint_violated = bool(x >= -2.4 and x <= -2.2) or \
                                 bool(x >= -1.3 and x <= -1.1) or \
                                 bool(x >= 1.1 and x <= 1.3) or \
                                 bool(x >= 2.2 and x <= 2.4)
        return is_constraint_violated

    def is_constraint2_violated(self, state):
        """
        Constraint 2: if theta > 4 degrees
        """
        _, _, theta, _ = state
        theta_threshold = 4 * 2 * math.pi / 360
        is_constraint_violated = bool(theta > theta_threshold) or bool(theta < -theta_threshold)
        return is_constraint_violated

    def step(self, state, action):
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        next_state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        cost = np.zeros(self._num_constraints)
        if not self.is_constraint1_violated(next_state):
            cost[0] = 1.0
        # give +1 value if constraint2 is not violated
        if not self.is_constraint2_violated(next_state):
            cost[1] = 1.0
        if not done:
            reward = 1.0
        else:
            reward = 0.0

        return next_state, reward, cost

    def get_init_state(self):
        """Get a random starting position."""
        state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return state

    def is_state_valid(self, state):
        x, _, theta, _ = state
        is_state_invalid = bool(
            x < -4.8 or x > 4.8
            or theta < -0.418 or theta > 0.418
        )
        return not is_state_invalid

    def is_state_over_bounds(self, state):
        """Returns True if the current state is out of bounds, i.e. the current run is over. Returns
        False otherwise."""

        x, x_dot, theta, theta_dot = state
        return bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
