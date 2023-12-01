#!/usr/bin/env python
import sys, gym, time
import gym
import numpy as np
import ray
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from openaigymexample import train_gym_game
from openaigymexample import get_trainer
import os
#
# Test yourself as a learning agent! Pass environment name as a command-line argument, for example:
#
# python keyboard_agent.py SpaceInvadersNoFrameskip-v4
#

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.
ACTIONS = None

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause, ACTIONS
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action, ACTIONS
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

def rollout(env, iterations):
    global human_agent_action, human_wants_restart, human_sets_pause, SKIP_CONTROL
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            #print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        iterations.append([obser, a, r, done, info])
        total_reward += r
        window_still_open = env.render()
        if window_still_open==False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))

def humanTrainGame(environment):
    env = gym.make(environment)
    env.reset()

    if not hasattr(env.action_space, 'n'):
        raise Exception('Keyboard agent only supports discrete action spaces')
    global ACTIONS
    ACTIONS = env.action_space.n

    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release
    gameIters = []

    while 1:
        window_still_open = rollout(env, gameIters)
        if window_still_open==False: break
    print("Starting training")
    ray.init(ignore_reinit_error=True)
    path = os.path.join(os.getcwd(), "gameplay", environment)
    writeToJson(env, gameIters, path)
    algo, trainer = get_trainer("DQN")

    config = algo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["input"] = path
    config["input_evaluation"] = ["wis"]
    print("Starting training")
    agent = train_gym_game(trainer(config, env=environment), 10)
    print("Training has finished")
    return agent

def writeToJson(env, input_list, path):
    batch_builder = SampleBatchBuilder()  # or MultiAgentSampleBatchBuilder
    writer = JsonWriter(path)

    # You normally wouldn't want to manually create sample batches if a
    # simulator is available, but let's do it anyways for example purposes:
    #env = gym.make(game)

    # RLlib uses preprocessors to implement transforms such as one-hot encoding
    # and flattening of tuple and dict observations. For CartPole a no-op
    # preprocessor is used, but this may be relevant for more complex envs.
    prep = lambda x:x
    #prep = get_preprocessor(env.observation_space)(env.observation_space)
    #print("The preprocessor is", prep)

    eps_id = 0
    t = 0
    obs = env.reset()
    prev_action = np.zeros_like(0)
    prev_reward = 0
    for i in range(len(input_list)):
        new_obs, action, rew, done, info = tuple(input_list[i])
        batch_builder.add_values(
            t=t,
            eps_id=eps_id,
            agent_index=0,
            #obs=prep.transform(obs),
            obs = obs,
            actions=action,
            action_prob=1.0,  # put the true action probability here
            action_logp=0.0,
            rewards=rew,
            prev_actions=prev_action,
            prev_rewards=prev_reward,
            dones=done,
            infos=info,
            #new_obs=prep.transform(new_obs))
            new_obs=new_obs)
        obs = new_obs
        prev_action = action
        prev_reward = rew
        t += 1
        if input_list[i][4] == True:
            eps_id = eps_id + 1
            t = 0
            prev_action = np.zeros_like(0)
            prev_reward = 0
    writer.write(batch_builder.build_and_reset())

if __name__ == "__main__":
    #environment = 'MountainCar-v0' if len(sys.argv) == 1 else sys.argv[1]
    #gameIters = []
    environment = "MountainCar-v0"
    #trained_agent, sumstats = humanTrainGame(environment)