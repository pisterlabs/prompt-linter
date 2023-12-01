import numpy as np
import math

class CartPole():

    #This environment has been taken from openai gym (https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py) with small modifications to the code.
    
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02

       
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self._max_episode_steps=1000
        self.steps = 0
        self.n_actions = 3
        self.n_states = 5

    def step(self, action):
        
        self.steps += 1
        x, x_dot, theta, theta_dot,_ = self.state
        if action == 0:
            force = self.force_mag  
        elif action == 1:
            force = -self.force_mag
        else:
            force = 0

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot, 0)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.steps >= self._max_episode_steps
        )
        
        c = np.cos(self.state[3])
        s = np.sin(self.state[3])
        self.state = np.array(self.state)
        self.state[3] = s
        self.state[4] = c
        reward = c
        return self.state, reward, done, {}

    def reset(self):
        self.steps = 0
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(5,))
        self.steps_beyond_done = None
        c = np.cos(self.state[3])
        s = np.sin(self.state[3])
        self.state[3] = s
        self.state[4] = c
        return self.state
