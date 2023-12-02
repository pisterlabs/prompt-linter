import numpy as np
import time
from ple import PLE
from ple.games.flappybird import FlappyBird


class FlappyBirdEnv:
    def __init__(self, sleep_time=0.0):
        """
        Constructs an environment like the environments in OpenAI Gym's library.

        :param sleep_time: float - how long the env should pause between each action.  This argument is
            only used when env.display = True so you can actually see what is going on.  Otherwise the
            game rendering is so fast that it is hard to see.

        There is only 1 possible action in Flappy Bird.  The getActionSet method retrieves all
        possible actions.  Ordinarily a single action would be represented in a binary way, (0, 1),
        where 1 is when the action is taken.  Here however, PLE returns [119, None], where 119 is the
        action.
        """
        self.sleep_time = sleep_time
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=False)
        self.env.init()
        self.action_map = self.env.getActionSet()  # [119, None]
        self.env_name = "FlappyBird"

    def get_observation(self):
        """
        The game state returns a dictionary whose keys describe what each value represents.
        This method returns the values of this dictionary as a numpy array, which
        matches the convention from OpenAI's Gym library.

        Game state returns a dict of 8 dimensions or features:
        player_y, player_vel,
        next_pipe_dist_to_player, next_pipe_top_y, next_pipe_bottom_y,
        next_next_pipe_dist_to_player, next_next_pipe_top_y, next_next_pipe_bottom_y

        :return: numpy array of shape (8,) representing dimensions or features of the game state
        """
        obs = self.env.getGameState()
        return np.array(list(obs.values()))

    def step(self, action):
        """
        Take an action and return the next observed state, reward, and done condition.

        :param action: int - List index for self.action_map for the action to take.  This is
            always either 0 or 1 for this particular game.

        :return: next observed state (np array of shape (8,)), reward (float), done condition (bool)
        """
        action = self.action_map[action]  # retrieves the action from the key:value map
        time.sleep(self.sleep_time)       # sleeps - useful if display=True so you can actually see what's going on
        reward = self.env.act(action)     # calculates reward or fitness
        done = self.env.game_over()       # checks if the game is over (Flappy Bird only ends when you lose)
        obs = self.get_observation()
        return obs, reward, done

    def reset(self):
        """
        Resets the game's state and returns an observation from the reset state
        """
        self.env.reset_game()
        return self.get_observation()

    def set_display(self, boolean_value):
        """
        Changes the display condition, which determines whether or not to display the Pygame
        in a screen to let you view what is going on.

        :param boolean_value: either True or False
        """
        self.env.display_screen = boolean_value
