"""
This is my attempt to implement DQN on the PONG environment from OpenAI gym.
NOTE: This is the pausable version and currently DOESN'T SAVE RANDOM STATE so NOT REPRODUCIBLE

To force myself to get better at version control, I will develop it all in this one file instead
of making backups each time I change something.

I have a trello board https://trello.com/b/iQUDEFxL/dqn
and a github repo https://github.com/naimenz/DQN

Building on the general structure I used for Sarsa/REINFORCE-type algorithms, I'll write a class to hold
all the important bits and share parameters and stuff.

"""

import numpy as np # using numpy as sparingly as possible, mainly for random numbers but also some other things
import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt
import time
import copy

# set a seed for reproducibility during implementation
SEED = 42
torch.manual_seed(SEED)
# using the new recommended numpy random number routines
rng = np.random.default_rng(SEED)

# To start with, follow a very similar architecture to what they did for Atari games
# NOTE I'm going to extend the Module class here: I've never really extended classes before so this is
# a possible failure point
class QNet(nn.Module):
    """
    This class defines the Deep Q-Network that will be used to predict Q-values of PONG states.

    I have defined this OUTSIDE the main class. I'll hardcode the parameters for now.
    TODO: Don't hardcode the parameters and make it good.
    """
    def __init__(self, n_outputs):
        super(QNet, self).__init__()
        # defining the necessary layers to be used in forward()
        # note we have four frames in an input state
        # parameters are (ALMOST) the same as used in the original DQN paper
        # NOTE Because I have 80*80 images instead of 84 by 84, I will pad by 2 to
        # get the same shape moving forward
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8,8), stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4,4), stride=2)
        # NOTE TODO: figure out how exactly it converts from input to output channels; it's not multiplicative
        self.linear1 = nn.Linear(2592, 256)
        # NOTE NOTE: Pong has SIX legal actions but only TWO actually do anything useful
        self.linear2 = nn.Linear(256, n_outputs)

    # define how the net handles input
    def forward(self, x):
        conv1_relu = self.conv1(x).clamp(min=0)
        conv2_relu = self.conv2(conv1_relu).clamp(min=0)
        # flattening the output of the convs to be passed to linear
        # BUT I don't want to flatten the batch dimension, so I'll say start_dim=1
        flat = torch.flatten(conv2_relu, start_dim=1)
        # appling the linear layers 
        linear1_relu = self.linear1(flat).clamp(min=0)
        output = self.linear2(linear1_relu)
        # should be a (batch, n_acts) output of Q values
        return output

class Buffer():
    """
    Writing a class to store and sample experienced transitions.

    NOTE: I am completely rewriting this because it's just using too much memory.
    Instead of storing full transitions, I will store action, rewards, and terminal flags
    as before but store frames individually and reconstruct states as needed.
    This will be slightly more fiddly but be roughly 8x more efficient
    """

    def __init__(self, max_size, im_dim, state_depth, max_ep_len):
        self.max_size = max_size # maximum number of elements to store
        self.im_dim = im_dim # dimensions of each frame
        self.state_depth = state_depth # number of frames in a state
        # storing FRAMES instead of STATES
        # Actually we need to store enough to keep an entire extra episode because otherwise we could 
        # overwrite transitions in a current episode
        self.frame_size = max_size + max_ep_len
        # NOTE TEST: using torch.half datatype because we don't have many discrete values
        # and it uses a LOT of memory
        self.frame_tensor = torch.empty(size=(self.frame_size,) + im_dim, dtype=torch.half)

        self.a_tensor = torch.LongTensor(size=(max_size,))
        self.r_tensor = torch.empty(size=(max_size,), dtype=torch.float)
        self.d_tensor = torch.empty(size=(max_size,), dtype=torch.bool)
        
        # keep track of which frames we need to construct the states
        self.s_ix_tensor = torch.LongTensor(size=(max_size, state_depth))
        self.sp_ix_tensor = torch.LongTensor(size=(max_size, state_depth))

        # I will keep track of the next index to insert at
        # Note that because this doesn't actually keep track of the state of
        # the buffer, it'll be internal and you should call .count() to see
        # how full it is
        self._counter = 0 

        # because the counter will loop, to know how many experiences we have 
        # i need to know if we've gone round already
        self.filled = False 

    # add a transition to the buffer
    # We pass the state, the action, the reward, the next state, and then three extras:
    # done: is sp terminal?
    # t: time from start of episode (if it's greater than state_depth then it doesn't matter)
    # fcount: the frame of training we are on currently
    def add(self, transition):
        s, a, r, sp, done, t, fcount = transition
        # the index of a, r, and done tensors to fill up, and what we aim to reconstruct
        counter = self._counter
        # writing to the easy buffers
        self.a_tensor[counter] = a
        self.r_tensor[counter] = r
        self.d_tensor[counter] = done
        # handling the hard frame buffer differently depending on if this is the initial state
        # if this is the first transition, we need to store frames for initial obs and first step
        # these will be the LAST two frames of sp
        if t == 0:
            # NOTE: This only works because the done flag tells us if we care about the sp state
            # and if we don't it doesn't matter what's in the buffer there so we can overwrite fcount
            self.frame_tensor[fcount % self.frame_size] = s[-1]
            self.frame_tensor[(fcount + 1) % self.frame_size] = sp[-1]
        # if this isn't the first transition, we only need to store the new frame sp[-1]
        else:
            self.frame_tensor[(fcount + 1) % self.frame_size] = sp[-1]

        # NOW we have to construct the tuples that tell us which frames to access
        # if we are near the beginning of the episode, we need multiple copies of the first frame
        s_ix = []
        if t < self.state_depth - 1:
            # first frame will be fcount - t
            first_frame_ix = fcount - t
            # we need state_depth - t of these 
            s_ix += [first_frame_ix] * (self.state_depth - t)
            # we'll need the t next frames as well
            for i in range(1, t+1):
                s_ix.append(first_frame_ix + i)
        # otherwise we just need the 'state_depth' states from fcount-self.state_depth+1 up to 
        # and including fcount
        else:
            for i in range(1, self.state_depth+1):
                s_ix.append(fcount - self.state_depth + i)
        # in both cases, sp_ix is just all but the first of s_ix with fcount + 1 o nthe front
        sp_ix = s_ix[1:] + [fcount+1]
        self.s_ix_tensor[counter] = torch.tensor(s_ix, dtype=torch.long)
        self.sp_ix_tensor[counter] = torch.tensor(sp_ix, dtype=torch.long)

        self._counter += 1
        # handle wrap-around
        if self._counter == self.max_size:
            self._counter = 0
            self.filled = True

    # get how many elements are in the buffer
    def count(self):
        # if we have filled already, then return max_size
        if self.filled:
            return self.max_size
        # else counter hasn't wrapped around yet so return it instead
        else:
            return self._counter

    # sample a random batch of experiences
    # NOTE this is tricky now because I need to reconstruct the states
    def sample(self, batch_size):
        # largest index to consider
        max_ix = self.count()
        # sample batch random indices 
        indices = torch.randint(low=0, high=max_ix, size=(batch_size,))
        a_sample = self.a_tensor[indices]
        r_sample = self.r_tensor[indices]
        d_sample = self.d_tensor[indices]
        # now I need to reconstruct s and sp
        s_ix_sample = self.s_ix_tensor[indices]
        sp_ix_sample = self.sp_ix_tensor[indices]
        # get the corresponding frames
        ftens = self.frame_tensor
        # TODO: make this not look so ABSOLUTELY horrible
        # convert to float32
        s_sample = torch.stack([torch.stack([ftens[i % self.frame_size] for i in ix]) for ix in s_ix_sample]).float()
        sp_sample = torch.stack([torch.stack([ftens[i % self.frame_size] for i in ix]) for ix in sp_ix_sample]).float()

        return s_sample, a_sample, r_sample, sp_sample, d_sample

class DQN():
    """
    DQN class specifically for solving Pong
    Might work on other Atari continuous obs, discrete act environments too if it works on Pong.
    """
    # we initialise with an environment for now, might need to add architecture once
    # I get the NN working.
    def __init__(self, env, gamma, eval_eps):
        self.eval_eps = eval_eps # epsilon to be used at evaluation time (typically lower than training eps)
        self.env = env
        self.gamma = gamma # discount rate (I think they used 0.99)
        # NOTE TEST: replacing the in-built actions with just THREE(now including noop) that are actually useful in Pong
        # NOTE TEST: adding an action set to index
        self.action_set = [0,2,5]
        self.n_acts = len(self.action_set)
        # self.n_acts = env.action_space.n # get the number of discrete actions possible

        # NOTE TEST: setting a maximum episode length of 5000 (will print episode lengths as I go though)
        # Episodes are unbounded but hopefully 10000 is a reasonable length (takes a lot of memory)
        self.max_ep_len = 10000
        # function to convert to greyscale (takes np frames)
        self.to_greyscale = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 

        # quickly get the dimensions of the images we will be using
        x = env.reset()
        self.im_dim = x.shape # this is before processing
        self.processed_dim = self.preprocess_frame(x).shape
        self.state_depth = 4 # number of frames in a state, hardcoded for now
        self.state_dim = (self.state_depth,) + self.processed_dim # we will use 4 most recent frames as the states

        # initialise the network TODO pass in params
        # now passing in n_acts at least
        self.qnet = self.initialise_network(n_outputs=self.n_acts)
    
    # Function to initialise the convolutional NN used to predict Q values
    def initialise_network(self, n_outputs):
        """
         TODO:
         - Make this more useful - i.e. calculate the values to pass in to the network
        """
        return QNet(n_outputs=n_outputs)

    # takes a batch of states of shape (batch,)+ self.state_dim as input and returns Q values as outputs
    # simple wrapper really
    def compute_Qs(self, s):
        return self.qnet(s)
    
    # get action for a state based on a given eps value
    # NOTE does not work with batches of states
    def get_act(self, s, eps):
        # 1 - eps of the time we pick the action with highest Q
        if rng.uniform() > eps:
            Qs = self.compute_Qs(s.unsqueeze(0)) # we have to add a batch dim
            act = torch.argmax(Qs)
        # rest of the time we just pick randomly among the n_acts actions
        else:
            act = torch.randint(0, self.n_acts, (1,))
        return act

    # preprocess a frame for use in constructing states
    # NOTE TEST: shifting color values so that the background is 0
    def preprocess_frame(self, frame):
        # greyscale is actually best done with brightness 
        frame = self.to_greyscale(frame)
        # Now I want to downsample the image in both dimensions
        # NOTE: maybe this isn't the best way to downsample
        frame = frame[::2, ::2]
        # Trying a crop because a lot of the vertical space is unused
        frame = frame[17:97, :]
        # NOTE TEST: subtracting background grey colour (87.258)
        frame = frame - 87.258
        # I'm going to rescale so that min and max values are at most 1 apart
        frame = frame / 255
        return torch.as_tensor(frame, dtype=torch.float)

    # encode the observation into a state based on the previous state and current obs
    def get_phi(self, s, obs):
        processed_frame = self.preprocess_frame(obs)
        # bump oldest frame and add new one
        # we do this by dropping the first element of s and concatenating the new frame
        # note that we have to unsqueeze the new frame so it has the same dimensions as s
        sp = torch.cat((s[1:], processed_frame.unsqueeze(0)))
        return sp
    
    # create an initial state by stacking four (or self.state_depth) copies of the first frame
    def initial_phi(self, obs):
        f = self.preprocess_frame(obs).unsqueeze(0)
        s = torch.cat(self.state_depth * (f,))
        return s

    # run an episode in evaluation mode 
    # NOTE: render accepts a float for time to sleep on each frame
    def evaluate(self, render=0):
        # alias env
        env = copy.deepcopy(self.env)
        
        # lists for logging
        ep_states = []
        ep_acts = []
        ep_rews = []

        # reset environment
        done = False
        obs = env.reset()
        # because we only have one frame so far, just make the initial state 4 copies of it
        s = self.initial_phi(obs)

        # loop over steps in the episode
        while not done:
            if render:
                env.render()
                time.sleep(render)
            act = self.get_act(s, self.eval_eps) # returns a 1-element tensor
            # NOTE TEST: converting an action in (0,1,2) into 0,2,5 (stay still, up and down in atari)
            av = self.action_set[act]
            obs, reward, done, info = env.step(av) 

            # log state, act, reward
            ep_states.append(s)
            ep_acts.append(act.item())
            ep_rews.append(reward)

            # construct new state from new obs
            s = self.get_phi(s, obs)

        return ep_states, ep_acts, ep_rews

    # generate a set of holdout states randomly for use as validation
    def generate_holdout(self, N):
        # we will store N states, each of size state_dim
        states = torch.empty(size=(N,) + self.state_dim, dtype=torch.float)

        t = 0 # frame counter
        done = True # indicate that we should restart episode immediately

        # alias env
        env = copy.deepcopy(self.env)
        
        # while we haven't seen enough frames
        while t < N:
            if done: # reset environment for a new episode
                done = False
                obs = env.reset()
                # because we only have one frame so far, just make the initial state 4 copies of it
                s = self.initial_phi(obs)

            # save the state
            states[t] = s
            # generate a random action given the current state
            act = self.get_act(s, 1.)
            # NOTE TEST: converting an action in (0,1,2) into 0,2,5 (stay still, up and down in atari)
            av = self.action_set[act]
            # act in the environment
            obs, reward, done, _ = env.step(av)

            # get the next state
            s = self.get_phi(s, obs)
            t += 1
        return states

    # evaluate the current Q function on a set of holdout states
    # we return the mean of the maximum Q for each state
    def evaluate_holdout(self, holdout):
        Qmax = torch.max(self.compute_Qs(holdout), dim=1)[0]
        return torch.mean(Qmax)


    def rets_from_rews(self, ep_rews, gamma):
        T = len(ep_rews) 
        rets = torch.tensor(ep_rews, dtype=torch.float)
        for i in reversed(range(T)):
            if i < T-1:
                rets[i] += gamma*rets[i+1]
        # return for final timestep is just 0
        return rets

    # evaluate current Q function on n episodes with self.eval_eps randomness
    # NOTE: Gives DISCOUNTED return
    def evaluate_on_n(self, n):
        # collect returns
        rets = []
        for i in range(n):
            _, _, ep_rews = self.evaluate(render=0)
            eval_rets = self.rets_from_rews(ep_rews, self.gamma)
            rets.append(eval_rets[0]) # episode return is return from t=0
        return np.mean(rets) # give mean return


    # functions to save and load a model
    def save_params(self, filename):
        s_dict = self.qnet.state_dict()
        torch.save(s_dict, filename)
        print(f"Parameters saved to {filename}")

    def load_params(self, filename):
        s_dict = torch.load(filename)
        self.qnet.load_state_dict(s_dict)
        print(f"Parameters loaded from {filename}")

    # compute loss on a batch of transitions
    # gradient of this should be what we need
    def compute_loss(self, s, a, r, sp, d):
        # don't need gradients except for Q(s,a)
        all_q = self.compute_Qs(s)
        q = all_q[range(len(s)), a]
        with torch.no_grad():
            # get the 'values' part of the max function and drop the 'indices'
            # NOTE TEST LOG: recording the maximising actions
            max_tuple = torch.max(self.compute_Qs(sp), dim=1)
            # print("Maximising indices",max_tuple[1])
            qsp = max_tuple[0]
            # setting terminal states to 0
            qsp[d] = 0
            # getting Q values for actions
            targets = r + self.gamma * qsp - q
        # loss is mean of targets * Q
        loss = -torch.mean(targets * q)
        return loss

    # given a minibatch of transitions, compute the sample gradient and take
    # the step
    def update_minibatch(self, minibatch, optim):
        optim.zero_grad()
        loss = self.compute_loss(*minibatch)
        loss.backward()
        optim.step()

    # NOTE: new function to train from a state dict so I can pause and resume training
    # Accepts no arguments beyond the state: does however assume that the dqn object has
    # been initialised in the same way
    def train_from_state(self, state):
        # unpack all the things 
        # fixed throughout the run
        directory = state['directory']
        n_evals = state['n_evals']
        N = state['total_frames']
        eps_epoch = state['eps_epoch']
        eps0 = state['eps0']
        eps1 = state['eps1']
        holdout = state['holdout'].float()
        lr = state['lr']
        n_eval_eps = state['n_eval_eps']
        # variable
        env_state = state['env_state']
        t = state['current_time']
        ep_t = state['episode_time']
        s = state['current_state']
        done = state['done']
        optim_state = state['optim_state']
        model_params = state['model_params']
        buf = state['buffer']
        total_time_elapsed = state['total_time_elapsed']
        # batch statistics
        batch_time_elapsed = state['batch_time_elapsed']
        holdout_scores = state['holdout_scores']
        recent_eps = state['recent_eps']
        ep_rets = state['ep_rets'] # history of episode returns
        ep_ret = state['ep_ret'] # CURRENT return being accumulated
        eval_scores = state['eval_scores'] # List of batch scores on n_eval_eps episodes

        # load in model parameters
        self.qnet.load_state_dict(model_params)
        # load in environment state (NOTE TODO: ASSUMES PONG frameskip=4 for now)
        env = copy.deepcopy(self.env)
        env.restore_full_state(env_state)
        # initialise and load optimiser state
        optim = torch.optim.Adam(self.qnet.parameters(), lr=lr)
        optim.load_state_dict(optim_state)

        # printing important variables at the start
        print(f"""\
============================================================================                
Resuming training.
Completed {t}/{N} frames (working on batch {1+(n_evals * t) // N}/{n_evals}).
Learning rate {lr}, buffer size {buf.max_size}, holdout size {len(holdout)}.

Current time elapsed: {total_time_elapsed:0.4f} seconds.
Current batch ran for: {batch_time_elapsed:0.4f} seconds.
============================================================================                
""")

        # SETTING UP NEW TIMERS
        tic = time.perf_counter() # batch timer
        bigtic = time.perf_counter()# total timer

        # CONSTRUCT EPSILON 
        epstep = (eps1 - eps0)/eps_epoch # the quantity to add to eps every frame
        get_eps = lambda t: eps0 + t*epstep if t < eps_epoch else eps1

        # MAIN TRAINING LOOP IN A TRY EXCEPT FOR CANCELLING
        while t < N:
            try:
            # while we haven't seen enough frames
                # NOTE LOG: evaluating 50 times throughout training
                if n_evals*t % N == 0 and t > 0:
                    print(" *** EVALUATING, DO NOT EXIT *** ")
                    liltic = time.perf_counter()
                    h_score = self.evaluate_holdout(holdout)
                    liltoc = time.perf_counter()
                    holdout_scores.append(h_score)
                    toc = time.perf_counter()

                    # NOTE TEST: evaluating the agent on n_eval_eps (10) episodes
                    etic = time.perf_counter()
                    eval_score = self.evaluate_on_n(n_eval_eps)
                    etoc = time.perf_counter()
                    print(f"Evaluation on {n_eval_eps} episodes took {etoc - etic:0.4f} seconds")
                    eval_scores.append(eval_score) # add it to the list

                    print(
                    f"""============== FRAME {t}/{N} (batch {(n_evals * t) // N}/{n_evals}) ============== 
Last {N/n_evals} frames took {batch_time_elapsed + toc - tic:0.4f} seconds.
Mean of recent episodes is {np.mean(ep_rets[recent_eps:])}.
Score on holdout is {h_score}.
Evaluation score on {n_eval_eps} episodes is {eval_score}.
                    """)
                    batch_time_elapsed = 0 # resetting time elapsed

                    # set new recent eps threshold
                    recent_eps = len(ep_rets)
                    # NOTE LOG saving the stats so far 
                    if not directory is None:
                        np.save(f"{directory}/DQNrets.npy", np.array(ep_rets))
                        np.save(f"{directory}/DQNh_scores.npy", np.array(holdout_scores))
                        np.save(f"{directory}/DQNeval_scores.npy", np.array(eval_scores))
                        # NOTE LOG I will overwrite parameters each time because they are big
                        self.save_params(f"{directory}/DQNparams.dat")
                        # save parameters separately 10 times
                        if 10*t % N == 0:
                            self.save_params(f"{directory}/{t}DQNparams.dat")
                    tic = time.perf_counter()

                if done: # reset environment for a new episode
                    # NOTE LOG: tracking episode return
                    if t > 0: # if this isn't the first episode
                        ep_rets.append(ep_ret)
                        # NOTE LOG: printing the length of the previous episode
                        print(f"Episode {len(ep_rets)} had length {ep_t}")
                    ep_ret = 0

                    # NOTE tracking episode time 
                    ep_t = 0

                    done = False
                    obs = env.reset()
                    # because we only have one frame so far, just make the initial state 4 copies of it
                    s = self.initial_phi(obs)

                # generate an action given the current state
                eps = get_eps(t)
                act = self.get_act(s, eps)
                # NOTE TEST: converting an action in (0,1,2) into 0,2,5 (stay still, up and down in atari)
                av = self.action_set[act]

                # act in the environment
                obsp, reward, done, _ = env.step(av)

                # NOTE LOG: tracking episode return
                ep_ret = ep_ret + (self.gamma**ep_t) * reward

                # get the next state
                sp = self.get_phi(s, obsp)

                # add all this to the experience buffer
                # PLUS the done flag so I know if sp is terminal
                # AND the various times
                buf.add((s, act, reward, sp, done, ep_t, t))

                # NOW WE SAMPLE A MINIBATCH and update on that
                minibatch = buf.sample(batch_size=32)
                self.update_minibatch(minibatch, optim)

                # prepare for next frame
                t += 1
                ep_t += 1 # updating the episode time as well
                s = sp

            except (KeyboardInterrupt, SystemExit):
                input("Press Ctrl-C again to exit WITHOUT saving or enter to save")
                print(f"\nSaving into {directory}/saved_state.tar")
                # ENDING TIMERS
                toc = time.perf_counter() # batch timer
                bigtoc = time.perf_counter() # total timer

                # GET MODEL PARAMETERS
                self.qnet.state_dict(model_params)

                print("TIME BEFORE SAVE:",t)
                # WRITING VARIABLE BITS TO STATE DICT
                state['env_state'] = env.clone_full_state()
                state['current_time'] = t
                state['episode_time'] = ep_t
                state['current_state'] = s
                state['done'] = done
                state['optim_state'] = optim.state_dict()
                state['model_params'] = model_params
                state['buffer'] = buf
                state['total_time_elapsed'] = total_time_elapsed + (bigtoc - bigtic)
                # batch statistics
                state['batch_time_elapsed'] = batch_time_elapsed + (toc - tic)
                state['holdout_scores'] = holdout_scores
                state['recent_eps'] = recent_eps
                state['ep_rets'] = ep_rets
                state['ep_ret'] = ep_ret # CURRENT return being accumulated
                state['eval_scores'] = eval_scores

                # WRITING STATE DICT TO FILE         
                torch.save(state, f"{directory}/saved_state.tar")
                # WRITING PAUSE MESSAGE TO info.txt
                pause_message = f"""\
Training paused at frame {t}/{N}.
Learning rate {lr}, buffer size {buf.max_size}, holdout size {len(holdout)}.
Time elapsed: {state['total_time_elapsed']:0.4f} seconds.
"""
                with open(f"{directory}/info.txt", 'w') as f: 
                   f.write(pause_message)

                # Holding so that cancelling is possible
                input("Press Ctrl-C again to exit or enter to continue")

        # AFTER WHILE LOOP AND TRY EXCEPT
        bigtoc = time.perf_counter()
        print(f"ALL TRAINING took {bigtoc - bigtic:0.4f} seconds")
        # NOTE LOG: tracking episode returns
        return ep_rets, holdout_scores

    # function to create a State dictionary to be used in training
    def initialise_training_state(self, N, lr, n_holdout, n_eval_eps, directory):
        # CALCULATING INITIAL VALUES
        tenth_N = int(N/10)
        buf_size = tenth_N
        eps_epoch = tenth_N
        # NOTE: episodes can go a lot longer in Pong
        buf = Buffer(max_size=buf_size, im_dim=self.processed_dim, state_depth=self.state_depth, max_ep_len=self.max_ep_len)
        tic = time.perf_counter()
        holdout = self.generate_holdout(N=n_holdout)
        toc = time.perf_counter()
        print(f"Generating holdout took {toc - tic:0.4f} seconds")

        state = dict()
        # fixed throughout run
        state['directory'] = directory 
        state['n_evals'] = 50 # HARDCODED for now
        state['total_frames'] = N 
        state['eps_epoch'] = eps_epoch 
        state['eps0'] = 1. # HARDCODED for now
        state['eps1'] = 0.1 # HARDCODED for now
        state['holdout'] = holdout.half()
        state['lr'] = lr 
        state['n_eval_eps'] = n_eval_eps # HARDCODED for now
        # variable
        state['env_state'] = self.env.clone_full_state() # CLONING state rather than env itself
        state['current_time'] = 0 
        state['episode_time'] = 0
        state['current_state'] = None # initially we have no state
        state['done'] = True # done set to True so we will get an initial state
        state['optim_state'] = torch.optim.Adam(self.qnet.parameters(), lr=lr).state_dict() # initialise the optimiser as Adam for now 
        state['model_params'] = self.qnet.state_dict()
        state['buffer'] = buf 
        state['total_time_elapsed'] = 0
        # batch statistics
        state['batch_time_elapsed'] = 0
        state['holdout_scores'] = []
        state['recent_eps'] = 0
        state['ep_rets'] = [] 
        state['ep_ret'] = 0
        state['eval_scores'] = []

        return state

    def train(self, N, lr, n_holdout, n_eval_eps, directory):
        # get an initial state dict
        state = self.initialise_training_state(N, lr, n_holdout, n_eval_eps, directory)
        # train with it
        ep_rets, holdout_scores = self.train_from_state(state)
        return ep_rets, holdout_scores
