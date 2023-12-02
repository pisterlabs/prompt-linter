import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from oracles._abstract import AbstractOracle
from dataset.rl.lunar_lander_dataset import LunarLanderDataset


class LunarLanderOracle(AbstractOracle):
    """
    Data generating process where the 8D input represents the state (position, orientation, velocity, leg contact)
    of an agent in the LunarLanderContinuous-v2 environment from OpenAI Gym. The reward function is modified
    so that after the agent successfully lands for a target number of timesteps, it needs to take off again
    and hover at a stable height.

    Input: x_position, y_position, x_velocity, y_velocity, rotation,
           rotational_velocity, left_contact, right_contact
    Internal state: num_land_steps [0, land_duration]
    Reward: hover_reward(instance) if num_land_steps >= land_duration else landing_reward(instance)
    """

    name = "lunarlander"
    input_shape = (8,)
    input_names = ["x_position", "y_position", "x_velocity", "y_velocity", "rotation",
                   "rotational_velocity", "left_contact", "right_contact"]
    internal_state_shape = (1,)

    land_duration = 50
    land_pad_width = 0.2

    def init_internal_state(self):
        return np.zeros(self.internal_state_shape, dtype=int)

    def _in_hover_zone(self, instance):
        return (-0.5 <= instance[0] <= 0.5) and (0.75 <= instance[1] <= 1.25)

    def _on_pad(self, instance):
        return -self.land_pad_width <= instance[0] <= self.land_pad_width \
               and instance[-2] > 0.5 and instance[-1] > 0.5

    def update_internal_state(self, instance):
        if self._on_pad(instance):
            self.internal_state[0] = min(self.internal_state[0] + 1, self.land_duration)

    def calculate_reward(self, instance):
        return self._hover_reward(instance) if self.internal_state[0] >= self.land_duration \
          else self._landing_reward(instance)

    def _target_y_reward(self, instance, target_y):
        pos_x, pos_y, vel_x, vel_y, ang, vel_ang, _, _ = instance
        return 0.1 * np.maximum(0., 2. - (np.sqrt(pos_x**2 + (pos_y - target_y)**2) + \
               np.sqrt(vel_x**2 + vel_y**2) + np.abs(ang) + np.abs(vel_ang)))

    def _landing_reward(self, instance):
        return self._target_y_reward(instance, target_y=0.) + float(self._on_pad(instance))

    def _hover_reward(self, instance):
        _, _, _, _, _, _, left_contact, right_contact = instance
        return self._target_y_reward(instance, target_y=1.) + \
               float(left_contact < 0.5 and right_contact < 0.5) + \
               float(self._in_hover_zone(instance))

    @classmethod
    def create_bags(cls, num_bags, min_bag_size, max_bag_size, seed, **kwargs):
        """
        Read episodes from the histories of RL agents trained on the ground-truth,
        then post-filter so that we end up with a specified ratio between the nine
        outcome types defined in LunarLanderDataset.generate_bag_metadata.
        """
        outcome_names = [
            "Pad never landed on",
            "Pad landed on; num steps on pad < 50; no take off",
            "Pad landed on; num steps on pad < 50; one or more take offs; in hover = 0",
            "Pad landed on; num steps on pad < 50; one or more take offs; 0 < in hover <= 20",
            "Pad landed on; num steps on pad < 50; one or more take offs; in hover > 20",
            "Pad landed on; num steps on pad >= 50; no take off",
            "Pad landed on; num steps on pad >= 50; one or more take off; in hover = 0",
            "Pad landed on; num steps on pad >= 50; one or more take off; 0 < in hover <= 20",
            "Pad landed on; num steps on pad >= 50; one or more take off; in hover > 20"
        ]
        bags, labels = [], []
        for fname in os.listdir(kwargs["load_path"]):
            if fname[-4:] == ".csv":
                print(fname)
                df = pd.read_csv(f'{kwargs["load_path"]}/{fname}')
                ep_starts = np.argwhere(df["time"].values == 0)[1:].flatten()
                bags += np.split(df[cls.input_names].values, ep_starts)
                labels += [r.sum() for r in np.split(df["reward"].values, ep_starts)]
        labels = np.array(labels)
        outcomes = np.array([LunarLanderDataset.generate_bag_metadata(bag)[1] for bag in bags])
        if kwargs["plot_outcomes"]:
            bins = np.linspace(labels.min(), labels.max(), 100)
            _, axes = plt.subplots(2, 5, sharex=True, sharey=True); axes = axes.flatten()
            axes[0].set_xlabel("Return Label"); axes[0].set_ylabel("Number of bags")
            for i, (ax, outcome_name) in enumerate(zip(axes, outcome_names)):
                labels_this_outcome = labels[np.argwhere(outcomes == i)]
                print(f"({i}) {outcome_name}: {len(labels_this_outcome)}")
                ax.hist(labels_this_outcome, bins=bins, color="k")
                ax.set_title(outcome_name.replace("; ", "\n"), fontsize=8)

        # =========================================================
        # NOTE: Selective reduction of outcome 8 to match outcome 4
        keep = np.ones(len(bags))
        outcome_4 = outcomes == 4
        outcome_8 = outcomes == 8
        print(outcome_4.sum(), outcome_8.sum())
        keep[np.random.choice(np.argwhere(outcome_8).flatten(), outcome_8.sum() - outcome_4.sum(), replace=False)] = 0
        bags = [bag for bag, k in zip(bags, keep) if k]
        print(len(bags))
        # =========================================================

        plt.show()

        return bags
