import gym
import numpy as np
from numpy.random import randint
from copy import deepcopy
import argparse

parser = argparse.ArgumentParser(description="generate data from openai gym's FrozenLake-v0")
parser.add_argument('--outfile',
			help="name of file to write output to; defaults to stdio",
			default=None)
parser.add_argument('--trials', type=int,
			help="number of trials to simulate; default 10",
            default=10)
parser.add_argument('--steps', type=int,
			help="number of decision steps to simulate for each trial; default 10",
            default=10)
parser.add_argument('--slip', type=bool,
			help="whether or not to add noise to transitions",
            default=True)
parser.add_argument('--noisy', type=bool,
			help="whether or not to add noise to observations",
            default=False)

args = parser.parse_args()

env = gym.make("FrozenLake-v0",is_slippery=args.slip)

from functools import reduce
out_array = [["Idnum"]+reduce(lambda x,y: x+y, [["observation"+str(i),"action"+str(i),"reward"+str(i)] for i in range(1,args.steps+1)])]

for i in range(args.trials):
    prev_state = 0
    env.reset()
    if args.outfile:
        log = [i]
    else:
        print("\ngame: "+str(i))
        env.render()
        print("[t,action,observation,reward]:")
    for t in range(args.steps):
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        if args.noisy:        
            observation = new_state if randint(1,101) <= 85 else prev_state
            prev_state = new_state
        else:
            observation = prev_state
        if args.outfile:
            log += [observation,action,reward]
        else:
            print([t,observation,action,reward])
            env.render()
    if args.outfile:
        out_array.append(log)

if args.outfile:
    import csv
    with open(args.outfile, 'w') as myfile:
    	wr = csv.writer(myfile,delimiter='\t')
    	for log in out_array:
    			wr.writerow(log)
