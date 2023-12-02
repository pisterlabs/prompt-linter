'''
This is the learn function from OpenAI's baselines.ppo1.pposgd_simple
rewritten with individual functions in .pposgd_simple_helper
OpenAI Baselines is licensed under the MIT License, see LICENSE
'''

from baselines.common.mpi_moments import mpi_moments
from baselines.ppo1.pposgd_simple import traj_segment_generator
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
from mpi4py import MPI
from .pposgd_simple_helper import build_policy_training_vars, build_counters, adjust_policy_learning_rate, update_policy, log_iter_info, calc_end_training

def learn(env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
        ):
    # Setup losses and stuff
    # ----------------------------------------

    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy

    loss_names, var_list, lossandgrad, adam, assign_old_eq_new, compute_losses = build_policy_training_vars(pi, oldpi, clip_param, entcoeff, adam_epsilon)
    mpi_moments_fn = lambda losses: mpi_moments(losses, axis=0)
    allgather_fn = MPI.COMM_WORLD.allgather

    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

    iters_so_far, episodes_so_far, timesteps_so_far, tstart, lenbuffer, rewbuffer = build_counters()

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        
        if calc_end_training(max_timesteps, timesteps_so_far,
                             max_episodes, episodes_so_far,
                             max_iters, iters_so_far,
                             max_seconds, tstart):
            break

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()

        cur_lrmult = adjust_policy_learning_rate(schedule, max_timesteps, timesteps_so_far, max_episodes, episodes_so_far, max_iters, iters_so_far)
        vpredbefore, tdlamret, optim_batchsize = update_policy(pi, seg, gamma, lam,
                                                     logger, optim_epochs, optim_batchsize, optim_stepsize, cur_lrmult,
                                                     loss_names, lossandgrad, adam, assign_old_eq_new, compute_losses,
                                                     mpi_moments_fn, allgather_fn)

        episodes_so_far, timesteps_so_far = log_iter_info(lenbuffer, rewbuffer, tstart,
                                                          vpredbefore, tdlamret, seg,
                                                          episodes_so_far, timesteps_so_far,
                                                          MPI.COMM_WORLD.Get_rank()==0)
        iters_so_far += 1

    return pi
