from argparse import ArgumentParser
import pickle
import json
import numpy as np
import gym
import minerl

from openai_vpt.agent import MineRLAgent

import logging
import coloredlogs
coloredlogs.install(logging.DEBUG)

def main(env, n_episodes=1, show=False):
    env = gym.make(env)

    seed = 4006
    env.seed(seed)
    env.reset()

    # Load expert actions
    with open(f"eval_seeds/demo-{seed}.json", "r") as f:
        expert_actions = json.load(f)

    for _ in range(n_episodes):
        env.seed(seed)
        obs = env.reset()
        
        # No action for a couple of steps
        for _ in range(40):
            no_act = env.action_space.noop()
            obs, reward, done, info = env.step(no_act)
            if show:
                env.render()

        for act in expert_actions:
            # Step your model here.
            # Currently, it's doing no actions
            # for 200 steps before quitting the episode
            no_act = env.action_space.noop()

            obs, reward, done, info = env.step(act)

            if show:
                env.render()
            if done:
                break
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    # parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    # parser.add_argument("--model", type=str, default="data/VPT-models/foundation-model-1x.model", help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, default="MineRLBasaltMakeWaterfall-v0")
    # parser.add_argument("--show", action="store_true", help="Render the environment.")

    args = parser.parse_args()

    main(args.env, show=True)
