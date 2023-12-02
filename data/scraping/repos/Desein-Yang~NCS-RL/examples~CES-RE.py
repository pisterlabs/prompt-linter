#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   CES-RE.py
@Time    :   2020/09/02 12:34:29
@Author  :   Qi Yang
@Version :   2.0
@Describtion:  CES paper official code + random embedding
'''
from src.optimizers import OpenAIOptimizer, CanonicalESOptimizer, CanonicalESMeanOptimizer
from src.policy import Policy
from src.logger import Logger

from argparse import ArgumentParser
from mpi4py import MPI
import numpy as np
import time
import json
import gym
from numpy.random import default_rng

def getRandomVector(self,ini_seed,idx,s):
    """Generate a [s,d] random vector with ini_seed.  
    Ini_seed is a grandpa seed to generate a [D] seed list."""
    rng = default_rng(ini_seed)
    child_seed = rng.integers(0,999999,size = (self.D,))
    rng = default_rng(child_seed[idx])
    x = rng.standard_normal((s,self.d))
    return x

def getSeed(lam,mu,cpus):
    """Get seeds list(seeds in a group is same).   
    E.g.[232913,232913,232913,345676,345676,345676,894356,894356,894356] 
    """
    seeds = np.zeros((cpus,),dtype = 'i')
    rng = np.random.RandomState(int(time.time()))
    for i in range(mu):
        s = rng.randint(999999) 
        seed = [s] * lam
        start = i * lam + 1
        seeds[start:start + lam] = seed
    return seeds


def from_y_to_x(y,param,seeds,group_id,D,s=10000,alpha=1.0):
    """Map effective params y (d-dimension) to x (D-dimension).  
        v1: x = A * y  
        v2: x' = ax + A*y
        v3: x'[j] = ax + A[j]*y 
    """
    tmp = np.empty((D,))
    assert s <= D
    if group_id is not None:
        for j in range(0,D,s):
            if j + s > D:
                s = D - 1 - j
            tmp[j:j+s,] = np.dot(getRandomVector(seeds[id_],j,s),y)
        x = alpha * param + tmp
    else:
        x = tmp
    return x

def getSeedPool(n_train,n_test,seed=None,zero_shot=True,range=1e5):
    """Create train and test random seed pool.  
    If use zero shot performance, test seed will non-repeatitive.  """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.default_rng()
    Allseed = np.arange(range,dtype=np.int)
    trainset = rng.choice(Allseed,size=n_train,replace=False)  # means sample without replacement
    tmp = np.setdiff1d(Allseed,trainset,assume_unique=True) # get disjoint set of trainset
    if zero_shot is False:
        testset = rng.choice(tmp,size=n_test,replace=False)
    else:
        testset = rng.choice(Allseed,size=n_test,replace=False)
    return trainset, testset

# This will allow us to create optimizer based on the string value from the configuration file.
# Add you optimizers to this dictionary.
optimizer_dict = {
    'OpenAIOptimizer': OpenAIOptimizer,
    'CanonicalESOptimizer': CanonicalESOptimizer,
    'CanonicalESMeanOptimizer': CanonicalESMeanOptimizer
}


# Main function that executes training loop.
# Population size is derived from the number of CPUs
# and the number of episodes per CPU.
# One CPU (id: 0) is used to evaluate currently proposed
# solution in each iteration.
# run_name comes useful when the same hyperparameters
# are evaluated multiple times.
def main(ep_per_cpu, game, configuration_file, run_name):
    start_time = time.time()

    with open(configuration_file, 'r') as f:
        configuration = json.loads(f.read())

    env_name = '%sNoFrameskip-v4' % game

    seed = 123456
    train_set, test_set = getSeedPool(10000,10000,seed)
    # MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpus = comm.Get_size()

    # One cpu (rank 0) will evaluate results
    train_cpus = cpus-1

    # Deduce population size
    lam = train_cpus * ep_per_cpu

    # Create environment
    env = gym.make(env_name)

    # Create policy (Deep Neural Network)
    # Internally it applies preprocessing to the environment state
    policy = Policy(env, network=configuration['network'], nonlin_name=configuration['nonlin_name'])

    # Create reference batch used for normalization
    # It will be overwritten with vb from worker with rank 0
    vb = policy.get_vb()

    # Extract vector with current parameters.
    parameters = policy.get_parameters()

    # Send parameters from worker 0 to all workers (MPI stuff)
    # to ensure that every worker starts in the same position
    comm.Bcast([parameters, MPI.FLOAT], root=0)
    comm.Bcast([vb, MPI.FLOAT], root=0)

    # Set the same virtual batch for each worker
    if rank != 0:
        policy.set_vb(vb)

    # Create optimizer with user defined settings (hyperparameters)
    OptimizerClass = optimizer_dict[configuration['optimizer']]
    optimizer = OptimizerClass(parameters, lam, rank, configuration["settings"])

    # Only rank 0 worker will log information from the training
    logger = None
    if rank == 0:
        # Initialize logger, save virtual batch and save some basic stuff at the beginning
        logger = Logger(optimizer.log_path(game, configuration['network'], run_name))
        logger.save_vb(vb)

        # Log basic stuff
        logger.log('Game'.ljust(25) + '%s' % game)
        logger.log('Network'.ljust(25) + '%s' % configuration['network'])
        logger.log('Optimizer'.ljust(25) + '%s' % configuration['optimizer'])
        logger.log('Number of CPUs'.ljust(25) + '%d' % cpus)
        logger.log('Population'.ljust(25) + '%d' % lam)
        logger.log('Dimensionality'.ljust(25) + '%d' % len(parameters))
        logger.log('Embedding Dim'.ljust(25) + '%d' % configuration['embed_dim'])
        logger.log('Seed pool?'.ljust(25) + '%d' % configuration['seed_pool'])
        # TODO:add configuration

        # Log basic info from the optimizer
        optimizer.log_basic(logger)

    # We will count number of steps
    # frames = 4 * steps (3 * steps for SpaceInvaders)
    steps_passed = 0
    while True:
        # Iteration start time
        iter_start_time = time.time()
        # Workers that run train episodes
        if rank != 0:
            # Empty arrays for each episode. We save: length, reward, noise index
            lens = [0] * ep_per_cpu
            rews = [0] * ep_per_cpu
            inds = [0] * ep_per_cpu

            # For each episode in this CPU we get new parameters,
            # update policy network and perform policy rollout
            for i in range(ep_per_cpu):
                ind, p = optimizer.get_parameters()
                policy.set_parameters(p)
                if configuration['seed_pool'] == 1:
                    env_seed = np.random.choice(train_set)
                else:
                    env_seed = np.random.randint(1e6)
                e_rew, e_len = policy.rollout(env_seed)
                lens[i] = e_len
                rews[i] = e_rew
                inds[i] = ind

            # Aggregate information, will later send it to each worker using MPI
            msg = np.array(rews + lens + inds, dtype=np.int32)

        # Worker rank 0 that runs evaluation episodes
        else:
            rews = [0] * ep_per_cpu
            lens = [0] * ep_per_cpu
            for i in range(ep_per_cpu):
                ind, p = optimizer.get_parameters()
                policy.set_parameters(p)

                env_seed = np.random.choice(test_set)
                e_rew, e_len = policy.rollout(env_seed)
                rews[i] = e_rew
                lens[i] = e_len

            eval_mean_rew = np.mean(rews)
            eval_max_rew = np.max(rews)

            # Empty array, evaluation results are not used for the update
            msg = np.zeros(3 * ep_per_cpu, dtype=np.int32)

        # MPI stuff
        # Initialize array which will be updated with information from all workers using MPI
        if rank ==0:
            logger.log('t1'.ljust(25) + '%f' % (time.time()-iter_start_time)) 
        results = np.empty((cpus, 3 * ep_per_cpu), dtype=np.int32)
        comm.Allgather([msg, MPI.INT], [results, MPI.INT])

        # Skip empty evaluation results from worker with id 0
        results = results[1:, :]

        # Extract IDs and rewards
        rews = results[:, :ep_per_cpu].flatten()
        lens = results[:, ep_per_cpu:(2*ep_per_cpu)].flatten()
        ids = results[:, (2*ep_per_cpu):].flatten()

        # Update parameters
        optimizer.update(ids=ids, rewards=rews)

        # Steps passed = Sum of episode steps from all offsprings
        steps = np.sum(lens)
        steps_passed += steps

        # Write some logs for this iteration
        # Using logs we are able to recover solution saved
        # after 1 hour of training or after 1 billion frames
        if rank == 0:
            iteration_time = (time.time() - iter_start_time)
            time_elapsed = (time.time() - start_time)/60
            train_mean_rew = np.mean(rews)
            train_max_rew = np.max(rews)
            logger.log('------------------------------------')
            logger.log('Iteration'.ljust(25) + '%f' % optimizer.iteration)
            logger.log('EvalMeanReward'.ljust(25) + '%f' % eval_mean_rew)
            logger.log('EvalMaxReward'.ljust(25) + '%f' % eval_max_rew)
            logger.log('TrainMeanReward'.ljust(25) + '%f' % train_mean_rew)
            logger.log('TrainMaxReward'.ljust(25) + '%f' % train_max_rew)
            logger.log('StepsSinceStart'.ljust(25) + '%f' % steps_passed)
            logger.log('StepsThisIter'.ljust(25) + '%f' % steps)
            logger.log('IterationTime'.ljust(25) + '%f' % iteration_time)
            logger.log('TimeSinceStart'.ljust(25) + '%f' % time_elapsed)

            # Give optimizer a chance to log its own stuff
            optimizer.log(logger)
            logger.log('------------------------------------')

            # Write stuff for training curve plot
            stat_string = "{},\t{},\t{},\t{},\t{},\t{}\n".\
                format(steps_passed, (time.time()-start_time),
                       eval_mean_rew, eval_max_rew, train_mean_rew, train_max_rew)
            logger.write_general_stat(stat_string)
            logger.write_optimizer_stat(optimizer.stat_string())

            # Save currently proposed solution every 20 iterations
            if optimizer.iteration % 20 == 1:
                logger.save_parameters(optimizer.parameters, optimizer.iteration)

        if steps_passed >= 25000000:
            print('finished')
            break

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-e', '--episodes_per_cpu',
                        help="Number of episode evaluations for each CPU, "
                             "population_size = episodes_per_cpu * Number of CPUs",
                        default=1, type=int)
    parser.add_argument('-g', '--game', help="Atari Game used to train an agent")
    parser.add_argument('-c', '--configuration_file', default = './config/CESpaper.json',help='Path to configuration file')
    parser.add_argument('-r', '--run_name', default='final', help='Name of the run, used to create log folder name', type=str)
    args = parser.parse_args()
    return args.episodes_per_cpu, args.game, args.configuration_file, args.run_name


if __name__ == '__main__':
    ep_per_cpu, game, configuration_file, run_name = parse_arguments()
    main(ep_per_cpu, game, configuration_file, run_name)
