import numpy as np
import time
import gym


class MsPacmanEnv:
    def __init__(self, sleep_time=0.0):
        """
        Constructs a wrapper for OpenAI Gym's Ms Pacman Atari environment.

        :param sleep_time: float - how long the env should pause between each action.  This argument is
            only used when env.display = True so you can actually see what is going on.  Otherwise the
            game rendering is so fast that it is hard to see.

        There is only 1 possible action in Flappy Bird.  The getActionSet method retrieves all
        possible actions.  Ordinarily a single action would be represented in a binary way, (0, 1),
        where 1 is when the action is taken.  Here however, PLE returns [119, None], where 119 is the
        action.
        """
        self.sleep_time = sleep_time
        self.env = gym.make('MsPacman-ram-v0')
        self.initial_obs = self.env.reset()
        self.action_map = [i for i in range(len(self.env.get_action_meanings()))]
        self.env_name = "MsPacman"

    def get_observation(self):
        """
        The game state returns a dictionary whose keys describe what each value represents.
        This method returns the values of this dictionary as a numpy array, which
        matches the convention from OpenAI's Gym library.

        Note:  This function should not be needed for OpenAI Gym, because the observation is returned
        by both the step() and reset() functions, but it is here to show how it could be obtained.

        :return: numpy array of shape (128,) representing dimensions or features of the game state
        """
        return self.env.ale.getRAM(np.zeros((self.env.ale.getRAMSize()), dtype=np.uint8))

    def step(self, action):
        """
        Take an action and return the next observed state, reward, and done condition.

        :param action: int - List index for self.action_map for the action to take.

        :return: next observed state (np array of shape (128,)), reward (float), done condition (bool)
        """
        time.sleep(self.sleep_time)  # sleeps - useful if display=True so you can actually see what's going on
        obs, reward, done = self.env.step(action)[:3]
        return obs, reward, done

    def reset(self):
        """
        Resets the game's state and returns an observation from the reset state

        :return: 1D numpy array of shape (128,) of the game state upon reset
        """
        return self.env.reset()

    def random_play(self):
        """
        Plays through a game taking random actions.
        """
        self.env.reset()
        episode_reward = 0
        while True:
            self.env.render()
            time.sleep(self.sleep_time)
            action = self.env.action_space.sample()
            state, reward, done, _ = self.env.step(action)
            episode_reward += reward
            if done:
                input("Press enter key to close game.")
                self.env.close()
                print("Total Reward:", episode_reward)
                break
