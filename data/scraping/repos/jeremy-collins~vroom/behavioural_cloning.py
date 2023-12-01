# Basic behavioural cloning
# Note: this uses gradient accumulation in batches of ones
#       to perform training.
#       This will fit inside even smaller GPUs (tested on 8GB one),
#       but is slow.

from argparse import ArgumentParser
import pickle
import time

import gym
import minerl
import torch as th
import numpy as np

from openai_vpt.agent import PI_HEAD_KWARGS, MineRLAgent
from data_loader import DataLoader
from openai_vpt.lib.tree_util import tree_map
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


# Originally this code was designed for a small dataset of ~20 demonstrations per task.
# The settings might not be the best for the full BASALT dataset (thousands of demonstrations).
# Use this flag to switch between the two settings
print("BC start")
USING_FULL_DATASET = True

EPOCHS = 1 if USING_FULL_DATASET else 2
# Needs to be <= number of videos
BATCH_SIZE = 64 if USING_FULL_DATASET else 16
# Ideally more than batch size to create
# variation in datasets (otherwise, you will
# get a bunch of consecutive samples)
# Decrease this (and batch_size) if you run out of memory
N_WORKERS = 100 if USING_FULL_DATASET else 20  #N_WORKERS = 100 #####
DEVICE = "cuda"

LOSS_REPORT_RATE = 1

# Tuned with bit of trial and error
LEARNING_RATE = 0.000181
# OpenAI VPT BC weight decay
# WEIGHT_DECAY = 0.039428
WEIGHT_DECAY = 0.0
# KL loss to the original model was not used in OpenAI VPT
KL_LOSS_WEIGHT = 1.0
MAX_GRAD_NORM = 5.0

MAX_BATCHES = 2000 if USING_FULL_DATASET else int(1e9)

def load_model_parameters(path_to_model_file):
    print("load_model_parameters")
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs

def behavioural_cloning_train(data_dir, in_model, in_weights, out_weights):
    print("behavioural_cloning_train")
    agent_policy_kwargs, agent_pi_head_kwargs = load_model_parameters(in_model)
    #print("agent_policy_kwargs: ", agent_policy_kwargs)
    #print("agent_pi_head_kwargs: ", agent_pi_head_kwargs)
    # To create model with the right environment.
    # All basalt environments have the same settings, so any of them works here
    print('gym.make("environment")')
    #env = gym.make("MineRLBasaltFindCave-v0") ####
    agent = MineRLAgent(device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    #agent.load_weights(in_weights)

    # Create a copy which will have the original parameters
    original_agent = MineRLAgent(device=DEVICE, policy_kwargs=agent_policy_kwargs, pi_head_kwargs=agent_pi_head_kwargs)
    #original_agent.load_weights(in_weights)
    #env.close() ####

    print("agent.policy")
    policy = agent.policy
    print("policy: ", policy)
    original_policy = original_agent.policy

    # Freeze most params if using small dataset
    for param in policy.parameters():
        param.requires_grad = False
    # Unfreeze final layers
    trainable_parameters = []
    for param in policy.net.lastlayer.parameters():
        param.requires_grad = True
        trainable_parameters.append(param)
    for param in policy.pi_head.parameters():
        param.requires_grad = True
        trainable_parameters.append(param)

    # Parameters taken from the OpenAI VPT paper
    optimizer = th.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    print("data_loader")
    print("data_dir: ", data_dir)
    data_loader = DataLoader(
        dataset_dir=data_dir,
        n_workers=N_WORKERS,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS,
    )

    start_time = time.time()

    # Keep track of the hidden state per episode/trajectory.
    # DataLoader provides unique id for each episode, which will
    # be different even for the same trajectory when it is loaded
    # up again
    episode_hidden_states = {}
    #print("episode_hidden_states: ", episode_hidden_states)
    dummy_first = th.from_numpy(np.array((False,))).to(DEVICE)

    loss_sum = 0
    for batch_i, (batch_images, batch_actions, batch_episode_id) in enumerate(data_loader):
        '''
        print("type(batch_images): ", type(batch_images))
        print("len(batch_images): ", len(batch_images))
        print("type(batch_actions): ", type(batch_actions))
        print("len(batch_actions): ", len(batch_actions))
        print("type(batch_episode_id): ", type(batch_episode_id))
        print("len(batch_episode_id): ", len(batch_episode_id))
        '''
        batch_loss = 0
        for image, action, episode_id in zip(batch_images, batch_actions, batch_episode_id):
            #print("type(image): ", type(image))
            #print("image.shape: ", image.shape)
            #print("len(action): ", len(action))
            #print("episode_id: ", episode_id)
            #print("action: ", action)
            #time.sleep(1000)
            #print("episode_hidden_states: ", episode_hidden_states)

            if image is None and action is None:
                # A work-item was done. Remove hidden state
                if episode_id in episode_hidden_states:
                    removed_hidden_state = episode_hidden_states.pop(episode_id)
                    #print("removed_hidden_state")
                    del removed_hidden_state
                continue
            #print("episode_hidden_states: ", episode_hidden_states)

            #agent_action = agent._env_action_to_agent(action, to_torch=True, check_if_null=True) #####
            agent_action = action
            #print("agent_action: ", agent_action)
            if agent_action is None:
                # Action was null
                continue

            agent_obs = agent._env_obs_to_agent({"pov": image}) # img 128x128x3 
            #print("agent_obs['img'].size(): ", agent_obs['img'].size())
            print("agent_obs['img']: ", agent_obs['img'])
            if episode_id not in episode_hidden_states:
                #print("agent_state = initial_state(1)")
                episode_hidden_states[episode_id] = policy.initial_state(1)
                #print("episode_hidden_states[episode_id]: ", episode_hidden_states[episode_id])
            agent_state = episode_hidden_states[episode_id]

            #print("agent_state: ", agent_state)
            #print("get_output_for_observation ")
            #agent_state =  th.tensor([.0,.0,.0,.0], device='cuda:0') 
            pi_distribution, _, new_agent_state = policy.get_output_for_observation(
                agent_obs, # img 128x128x3
                agent_state,
                dummy_first
            )

            with th.no_grad():
                original_pi_distribution, _, _ = original_policy.get_output_for_observation(
                    agent_obs,
                    agent_state,
                    dummy_first
                )

            print("agent_action: ", agent_action)
            print("vs")
            print("pi_distribution: ", pi_distribution)
            #time.sleep(1000)
            #print("new_agent_state: ", new_agent_state)

            #log_prob  = policy.get_logprob_of_action(pi_distribution, agent_action)
            log_prob =  th.tensor([-17.8694], device='cuda:0') #, grad_fn=<SelectBackward>)
            kl_div = policy.get_kl_of_action_dists(pi_distribution, original_pi_distribution)
            #kl_div =  th.tensor([0.7508], device='cuda:0')
            print("type(kl_div): ", type(kl_div))
            print("kl_div: ", kl_div)
            # Make sure we do not try to backprop through sequence
            # (fails with current accumulation)
            new_agent_state = tree_map(lambda x: x.detach(), new_agent_state)
            episode_hidden_states[episode_id] = new_agent_state

            # Finally, update the agent to increase the probability of the
            # taken action.
            # Remember to take mean over batch losses
            loss = (-log_prob + KL_LOSS_WEIGHT * kl_div) / BATCH_SIZE
            print("type(loss): ", type(loss))
            print("loss: ", loss)

            batch_loss += loss.item()
            loss.backward()

        th.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

        loss_sum += batch_loss
        if batch_i % LOSS_REPORT_RATE == 0:
            time_since_start = time.time() - start_time
            print(f"Time: {time_since_start:.2f}, Batches: {batch_i}, Avrg loss: {loss_sum / LOSS_REPORT_RATE:.4f}")
            writer.add_scalar("Loss", loss_sum / LOSS_REPORT_RATE, batch_i) #, "Avrg loss", loss_sum / LOSS_REPORT_RATE)
            
            loss_sum = 0

        if batch_i > MAX_BATCHES:
            break
    print("save model")
    state_dict = policy.state_dict()
    #print(state_dict)
    print(out_weights)
    th.save(state_dict, out_weights)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the directory containing recordings to be trained on")
    parser.add_argument("--in-model", required=True, type=str, help="Path to the .model file to be finetuned")
    parser.add_argument("--in-weights", required=True, type=str, help="Path to the .weights file to be finetuned")
    parser.add_argument("--out-weights", required=True, type=str, help="Path where finetuned weights will be saved")

    args = parser.parse_args()
    behavioural_cloning_train(args.data_dir, args.in_model, args.in_weights, args.out_weights)
