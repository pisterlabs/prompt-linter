import numpy as np

from oracles._abstract import AbstractOracle


class LunarLanderTimerOracle(AbstractOracle):
    """
    Data generating process where the 8D input represents the state (position, orientation, velocity, leg contact)
    of an agent in the LunarLander-v2 environment from OpenAI Gym. The reward function is modified
    so that the agent gets rewarded for remaining off the ground for the first half of the episode, then landing
    in the second half.

    Input: x_position, y_position, x_velocity, y_velocity, rotation,
           rotational_velocity, left_contact, right_contact
    Internal state: time [0, max_bag_size]
    Reward: (_hovering(instance) if time <= flip_time else _on_pad(instance))
    """

    name = "lunarlandertimer"
    input_shape = (8,)
    input_names = ["x_position", "y_position", "x_velocity", "y_velocity", "rotation",
                   "rotational_velocity", "left_contact", "right_contact"]
    internal_state_shape = (1,)

    flip_time = 150 # NOTE: Episodes should be double this to make rewards balanced
    land_pad_width = 1.0

    def init_internal_state(self):
        return np.zeros(self.internal_state_shape, dtype=int)

    def _hovering(self, instance):
        # NOTE: On the left
        return instance[0] < 0.0 and instance[6] < 0.5 and instance[7] < 0.5

    def _on_pad(self, instance):
        return (-self.land_pad_width <= instance[0] <= self.land_pad_width) \
               and instance[6] > 0.5 and instance[7] > 0.5

    def _shaping(self, instance):
        pos_x, pos_y, vel_x, vel_y, ang, vel_ang, _, _ = instance
        # return 0.1 * np.maximum(0., 2. - (np.sqrt(pos_x**2 + (pos_y - 0.)**2) + \
            #    np.sqrt(vel_x**2 + vel_y**2) + np.abs(ang) + np.abs(vel_ang)))
        return 0.1 * np.maximum(0., 2. - (np.sqrt(vel_x**2 + vel_y**2) + np.abs(ang) + np.abs(vel_ang)))

    def update_internal_state(self, instance):
        self.internal_state[0] += 1

    def calculate_reward(self, instance):
        return float(instance[0] <= 0.0) if (self.internal_state[0] <= self.flip_time) \
          else float(instance[0] >  0.0)

    # def calculate_reward(self, instance):
    #     return self._shaping(instance) + \
    #         (float(self._hovering(instance)) if (self.internal_state[0] <= self.flip_time) \
    #         else float(self._on_pad(instance)))

    @classmethod
    def create_bags(cls, num_bags, min_bag_size, max_bag_size, seed, **kwargs):
        pass
