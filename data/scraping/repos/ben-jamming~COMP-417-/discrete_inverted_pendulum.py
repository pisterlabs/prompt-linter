import pygame, sys
import numpy as np
from pygame.locals import *
import math
import RL_controller
import argparse
import time
import pygame.camera
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


# [-90, 90] x 20 -> 4.5
# p-5, 5] x 20

# Base engine from the following link
# https://github.com/the-lagrangian/inverted-pendulum
#See report for reward function source

class DiscreteInvertedPendulum(object):
    def __init__(self, args, windowdims, cartdims, penddims, action_range=[-1, 1]):
        self.args = args
        self.action_range = action_range

        self.window_width = windowdims[0]
        self.window_height = windowdims[1]

        self.cart_width = cartdims[0]
        self.car_height = cartdims[1]
        self.pendulum_width = penddims[0]
        self.pendulum_length = penddims[1]

        self.Y_CART = 3 * self.window_height / 4
        self.reset()
        if args.add_noise_to_gravity_and_mass:
            self.gravity = args.gravity + np.random.uniform(-5, 5)
            self.masscart = 1.0 + np.random.uniform(-0.5, 0.5)
            self.masspole = 0.1 + np.random.uniform(-0.05, 0.2)
        else:
            self.gravity = args.gravity
            self.masscart = 1.0
            self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.dt = args.dt  # seconds between state updates
        # Angle at which to fail the episode
        self.theta_threshold_radians =  math.pi / 2
        self.x_threshold = 2.4
        self.theta_dot_threshold = 7

        self.x_conversion = self.window_width / 2 / self.x_threshold


    def reset(self):
        """initializes pendulum in upright state with small perturbation"""
        self.terminal = False
        self.timestep = 0

        self.x_dot = np.random.uniform(-0.03, 0.03)
        self.x = np.random.uniform(-0.01, 0.01)

        self.theta = np.random.uniform(-0.03, 0.03)
        self.theta_dot = np.random.uniform(-0.01, 0.01)
        self.total_reward = 0
        self.reward = 0

    def get_reward(self, discrete_theta):
        #small survival reward + angle reward
        current_angle = self.from_discrete(discrete_theta, self.args.theta_discrete_steps,
                                                        range=[-math.pi/2, math.pi/2])
        print("theta:")
        print(current_angle)
        new_reward = 0
        if current_angle >= -0.1 and current_angle <= 0.1:
            new_reward = 1
        else:
            new_reward = 0

        if np.abs(current_angle - self.theta_threshold_radians) < 0.001:
            new_reward = 5
        return new_reward

        """return 0.00001 + 0.99999 * (self.theta_threshold_radians -
                              np.abs(self.from_discrete(discrete_theta, self.args.theta_discrete_steps,
                                                        range=[-math.pi/2, math.pi/2])))/self.theta_threshold_radians"""

    def to_discrete(self, value, steps, range):
        value = np.clip(value, range[0], range[1])         #Threshold it
        value = (value - range[0])/(range[1] - range[0])   #normalize to [0, 1]
        value = int(value * steps * 0.99999) #ensure it cannot be exactly steps
        return value

    def from_discrete(self, discrete_value, steps, range):
        value = (discrete_value + 0.5)/steps #on average the discrete value gets rounded down even if it was 19.99 -> 19 so we use +0.5 as more accurate
        value = value * (range[1] - range[0]) + range[0]
        return value

    def get_continuous_values(self):
        return (self.terminal, self.timestep, self.x,
                self.x_dot, self.theta, self.theta_dot, self.reward)

    def get_discrete_values(self):
        discrete_theta = self.to_discrete(self.theta, self.args.theta_discrete_steps, range=[-math.pi/2, math.pi/2])
        discrete_theta_dot = self.to_discrete(self.theta_dot, self.args.theta_dot_discrete_steps, range=[-self.theta_dot_threshold, self.theta_dot_threshold])

        return (self.terminal, self.timestep, discrete_theta, discrete_theta_dot, self.reward)

    def set_state(self, state):
        terminal, timestep, x, x_dot, theta, theta_dot = state
        self.terminal = terminal
        self.timestep = timestep
        self.x = x
        self.x_dot = x_dot
        self.theta = theta  # in radians
        self.theta_dot = theta_dot  # in radians

    def step(self, action):
        if action == 0:
            force = -10
        elif action == 1:
            force = 0
        elif action == 2:
            force = 10
        else:
            raise Exception("Invalid Action, Actions are only 0, 1, 2")

        # From OpenAI CartPole
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        self.timestep += 1

        costheta = math.cos(self.theta)
        sintheta = math.sin(self.theta)
        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
                       force + self.polemass_length * self.theta_dot ** 2 * sintheta
               ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        self.x = self.x + self.dt * self.x_dot
        self.x_dot = self.x_dot + self.dt * xacc
        self.theta = self.theta + self.dt * self.theta_dot
        self.theta_dot = self.theta_dot + self.dt * thetaacc

        self.terminal = bool(
            self.x < -self.x_threshold
            or self.x > self.x_threshold
            or self.theta < -self.theta_threshold_radians
            or self.theta > self.theta_threshold_radians
        )
        # radians to degrees
        # within -+ 15
        # if (self.theta * 57.2958) < 15 and (self.theta * 57.2958) > -15:
        #     self.score += 1
        #     self.reward = 1
        # else:
        #     self.reward = 0
        self.reward = self.get_reward(self.to_discrete(self.theta, self.args.theta_discrete_steps, range=[-math.pi/2, math.pi/2]))
        self.total_reward = self.total_reward + self.reward
        return self.get_continuous_values()


class InvertedPendulumGame(object):
    def __init__(self, args, windowdims=(800, 400), cartdims=(50, 10), penddims=(6.0, 150.0), refreshfreq=1000, mode=None):
        self.args = args
        self.RL_controller = mode
        self.max_timestep = args.max_timestep
        self.game_round_number = 0
        self.pendulum = DiscreteInvertedPendulum(args, windowdims, cartdims, penddims)
        self.performance_figure_path = args.performance_figure_path

        self.window_width = windowdims[0]
        self.window_height = windowdims[1]

        self.cart_width = cartdims[0]
        self.car_height = cartdims[1]
        self.pendulum_width = penddims[0]
        self.pendulum_length = penddims[1]
        self.manual_action_magnitude = args.manual_action_magnitude
        self.random_controller = args.random_controller
        self.noisy_actions = args.noisy_actions

        self.score_list = []

        self.Y_CART = self.pendulum.Y_CART
        # self.time gives time in frames
        self.timestep = 0

        pygame.init()
        self.clock = pygame.time.Clock()
        # specify number of frames / state updates per second
        self.REFRESHFREQ = refreshfreq
        self.surface = pygame.display.set_mode(windowdims, 0, 32)
        pygame.display.set_caption('Inverted Pendulum Game')
        #array specifying corners of pendulum to be drawn
        self.static_pendulum_array = np.array(
            [[-self.pendulum_width / 2, 0],
             [self.pendulum_width / 2, 0],
             [self.pendulum_width / 2, -self.pendulum_length],
             [-self.pendulum_width / 2, -self.pendulum_length]]).T
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.WHITE = (255, 255, 255)

    def draw_cart(self, x, theta):
        cart = pygame.Rect(
            self.pendulum.x * self.pendulum.x_conversion + self.pendulum.window_width / 2 - self.cart_width // 2,
            self.Y_CART, self.cart_width, self.car_height)
        pygame.draw.rect(self.surface, self.RED, cart)
        pendulum_array = np.dot(self.rotation_matrix(-theta), self.static_pendulum_array)
        pendulum_array += np.array([[x * self.pendulum.x_conversion + self.pendulum.window_width / 2], [self.Y_CART]])
        pendulum = pygame.draw.polygon(self.surface, self.BLACK,
                                       ((pendulum_array[0, 0], pendulum_array[1, 0]),
                                        (pendulum_array[0, 1], pendulum_array[1, 1]),
                                        (pendulum_array[0, 2], pendulum_array[1, 2]),
                                        (pendulum_array[0, 3], pendulum_array[1, 3])))

    @staticmethod
    def rotation_matrix(theta):
        return np.array([[np.cos(theta), np.sin(theta)],
                         [-1 * np.sin(theta), np.cos(theta)]])

    def render_text(self, text, point, position="center", fontsize=48):
        font = pygame.font.SysFont(None, fontsize)
        text_render = font.render(text, True, self.BLACK, self.WHITE)
        text_rect = text_render.get_rect()
        if position == "center":
            text_rect.center = point
        elif position == "topleft":
            text_rect.topleft = point
        self.surface.blit(text_render, text_rect)

    def time_seconds(self):
        return self.timestep / float(self.REFRESHFREQ)

    def starting_page(self):
        self.surface.fill(self.WHITE)
        self.render_text("Inverted Pendulum",
                         (0.5 * self.window_width, 0.4 * self.window_height))
        self.render_text("COMP 417 Assignment 2",
                         (0.5 * self.window_width, 0.5 * self.window_height),
                         fontsize=30)
        self.render_text("Press Enter to Begin",
                         (0.5 * self.window_width, 0.7 * self.window_height),
                         fontsize=30)
        pygame.display.update()

    def save_current_state_as_image(self, path):
        im = Image.fromarray(self.surface_array)
        im.save(path + "current_state.png")

    def game_round(self):
        LEFT = 0
        NO_ACTION = 1
        RIGHT = 2
        self.pendulum.reset()
        if not self.RL_controller is None:
            self.RL_controller.reset()

        theta_diff_list = []


        action = NO_ACTION
        for i in range(self.max_timestep):
            self.surface_array = pygame.surfarray.array3d(self.surface)
            self.surface_array = np.transpose(self.surface_array, [1, 0, 2])

            if self.RL_controller is None:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN:
                        if event.key == K_LEFT:
                            action = LEFT  # "Left"
                        if event.key == K_RIGHT:
                            action = RIGHT
                    if event.type == KEYUP:
                        if event.key == K_LEFT:
                            action = NO_ACTION
                        if event.key == K_RIGHT:
                            action = NO_ACTION
                        if event.key == K_ESCAPE:
                            pygame.quit()
                            sys.exit()
            else:
                action = self.RL_controller.get_action(self.pendulum.get_discrete_values(), self.surface_array,
                                                        random_controller=self.random_controller, episode=self.game_round_number)
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN:
                        if event.key == K_ESCAPE:
                            print("Exiting ... ")
                            pygame.quit()
                            sys.exit()

            if self.noisy_actions and self.RL_controller is None:
                action = action + np.random.uniform(-0.1, 0.1)

            terminal, timestep, x, _, theta, _, _ = self.pendulum.step(action)
            theta_diff_list.append(np.abs(theta))

            self.timestep = timestep
            self.surface.fill(self.WHITE)
            self.draw_cart(x, theta)

            time_text = "t = {}".format(int(self.pendulum.timestep))
            self.render_text(time_text, (0.1 * self.window_width, 0.1 * self.window_height),
                             position="topleft", fontsize=40)

            pygame.display.update()
            self.clock.tick(self.REFRESHFREQ)
            if terminal:
                break;

        self.game_round_number += 1

        if(self.game_round_number%20 == 0) or self.game_round_number==3000:
            plt.plot(np.arange(len(theta_diff_list)), theta_diff_list)
            plt.xlabel('Time')
            plt.ylabel('|Theta(radians)|')
            plt.title("|Theta| vs Time")
            plt.savefig(self.performance_figure_path + "_run_" + str(len(self.score_list)) + ".png")
            plt.close()
            self.RL_controller.save_state_matrix(self.game_round_number)

          

        self.score_list.append(int(self.pendulum.total_reward))

    def end_of_round(self):
        self.surface.fill(self.WHITE)
        self.draw_cart(self.pendulum.x, self.pendulum.theta)
        self.render_text("Score: {}".format(int(self.pendulum.total_reward)),
                         (0.5 * self.window_width, 0.3 * self.window_height))
        self.render_text("Average Score : {}".format(np.around(np.mean(self.score_list), 3)),
                         (0.5 * self.window_width, 0.4 * self.window_height))
        self.render_text("Standard Deviation Score : {}".format(np.around(np.std(self.score_list), 3)),
                         (0.5 * self.window_width, 0.5 * self.window_height))
        self.render_text("Runs : {}".format(len(self.score_list)),
                         (0.5 * self.window_width, 0.6 * self.window_height))
        if self.RL_controller is None:
            self.render_text("(Enter to play again, ESC to exit)",
                             (0.5 * self.window_width, 0.85 * self.window_height), fontsize=30)
        pygame.display.update()
        time.sleep(2.0)

    def game(self):
        self.starting_page()
        while True:
            if self.RL_controller is None:  # Manual mode engaged
                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN:
                        if event.key == K_RETURN:
                            self.game_round()
                            self.end_of_round()
                        if event.key == K_ESCAPE:
                            pygame.quit()
                            sys.exit()
            else:  # Use the PID controller instead, ignores input expect exit
                self.game_round()
                self.end_of_round()
                self.pendulum.reset()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="RL")
    parser.add_argument('--random_controller', type=bool, default=False)
    parser.add_argument('--add_noise_to_gravity_and_mass', type=bool, default=False)
    parser.add_argument('--max_timestep', type=int, default=3000)
    parser.add_argument('--dt', type=float, default=0.01)

    parser.add_argument('--gravity', type=float, default=9.81)
    parser.add_argument('--manual_action_magnitude', type=float, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noisy_actions', type=bool, default=False)
    parser.add_argument('--performance_figure_path', type=str, default="performance_figure")

    parser.add_argument('--theta_discrete_steps', type=int, default=40)
    parser.add_argument('--theta_dot_discrete_steps', type=int, default=40)

    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=0.05)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    np.random.seed(args.seed)
    if args.mode == "manual":
        inv = InvertedPendulumGame(args, mode=None)
    else:
        inv = InvertedPendulumGame(args, mode=RL_controller.RL_controller(args))
    inv.game()


if __name__ == '__main__':
    main()
    #print(148**2 % 421)
