#Code inspired by envs from openai-gym
import math
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from zipdl.utils import seeding
from zipdl.utils import utils
from zipdl.utils.spaces.discrete import Discrete
from zipdl.utils.spaces.db_node import DB2Node

from zipline import run_algorithm

START_CASH = 5000
#TODO: Allow partitioning of the time span of data to allow for cross-validation testing
TRADING_START = dt.strptime('2010-01-01', '%Y-%m-%d')
TRADING_END = dt.strptime('2016-01-01', '%Y-%m-%d')
ENV_METRICS = ['t3m', 'ps-1mdelat', 'vixD1m']
NUM_BUCKETS = [3, 3, 3]
#Where the first element is the starting factor weighting
FACTOR_WEIGHTS = [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3], [0.9, 0.1]]

#Save plots of reward
SAVE_FIGS=False
RENDER = False

if RENDER:
    plt.ion()

'''
MODEL OF NODES:
      
  [1,9]<->[3,7]<->[5,5]<->[7,3]<->[9,1]
    ---------------^ ^---------------
    *Each node is also able to remain stationary
    Ie. there are three moves for every node - left, stationary, and right.

Nodes action spaces parameterized as follows: [*, *, *, *, *]
Where entry corresponds to model above, and a 1 in the entry corresponds to a valid move. 
'''

class Dynamic_beta_env:
    def __init__(self, trading_start, rrn=False, num_frames=None):
        '''
        Default rebalancing window = 'monthly'
        algo is a tuple of the following functions:
        (
            initialize_intialize: a function to create an initialize function
                Should have parameters for window length and weights
            handle_data
            rebalance_portfolio
        )
        '''
        self.starting_cash = START_CASH

        self.rrn = rrn
        self.num_frames = num_frames
        self.current_node = self.initialize_nodes()
        self.action_space = Discrete(3)
        self.observation_space = dict({metric : Discrete(bucket_num) for metric, bucket_num in zip(ENV_METRICS, NUM_BUCKETS)})
        self.observation_space['Curr state'] = Discrete(len(DB2Node.Nodes2))
        self.start = trading_start
        curr = np.array([utils.get_metric_bucket(self.start, metric) for metric in ENV_METRICS] + [3])
        assert len(curr) == len(ENV_METRICS) + 1
        if self.rrn:
            self.state = deque(maxlen=num_frames)
            #print(state)
            self.state.append(curr)
        else:
            self.state = curr
        #self.state = np.reshape(self.state, [1, len(self.observation_space)])
        self.prev_state = None

        if RENDER:
            reset_render()
            
    def initialize_nodes(self):
        #Initialize nodes according to mdp, and return starting nodes
        counter = 0
        for weight in FACTOR_WEIGHTS:
            DB2Node(weight, counter)
            counter += 1
        return DB2Node.Nodes2[3] #Starting node of [5,5]

    def update_state(self, date):
        self.prev_state = self.state.copy()
        curr = np.array([utils.get_metric_bucket(date, metric) for metric in ENV_METRICS] + [self.current_node.id])
        print(date)
        assert len(curr) == len(ENV_METRICS) + 1
        if self.rrn:
            self.state.append(curr)
        else:
            self.state = curr #np.reshape(state, [1, len(self.observation_space)])
        return self.state

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        if action == 0: 
            next_index = (self.current_node.index - 1) % len(DB2Node.Nodes2)
        elif action == 2: 
            next_index = (self.current_node.index + 1) % len(DB2Node.Nodes2)
        else:
            next_index = self.current_node.index
        self.current_node = DB2Node.Nodes2[next_index]
        return self.current_node.weights

    def reset(self):
        '''
        Reconstruct initial state
        '''
        if SAVE_FIGS:
            self.viewer.savefig('ddqn{}'.format(self.counter))
            self.counter += 1
        if RENDER:
            reset_render()
        #self.current_node = DB2Node.Nodes2[3]
        metrics = np.array([utils.get_metric_bucket(self.start, metric) for metric in ENV_METRICS] + [3])
        if self.rrn:
            self.state.clear()
            self.state.append(metrics)
        else:
            self.state = metrics
        return np.array(self.state)

    def render(self):
        '''
        View the windowly rewards of each trial action
        ie. Default - view the sortino of each trial
        '''
        self.viewer.show()
    
    def reset_render(self):
        self.min_x = 0
        #self.max_x = (TRADING_END - TRADING_START - self.timedelta).days #ie. the max size of the training set
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([], [], 'o')
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        self.counter = 0

    def update_viewer(self, new_data):
        self.viewer.set_xdata(np.append(self.viewer.get_xdata(), self.counter))
        self.viewer.set_ydata(np.append(self.viewer.get_ydata(), new_data))
        self.ax.relim()
        self.ax.autoscale_view()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def close(self):
        if self.viewer: self.viewer.close()