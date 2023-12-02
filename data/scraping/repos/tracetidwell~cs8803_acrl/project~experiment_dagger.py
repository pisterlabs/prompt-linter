from comet_ml import Experiment

import os
import copy
import uuid
import time
import glob

import gym_compete
import gym

import numpy as np
import torch
from torch import from_numpy

from utils import ReplayBuffer
from logx import EpochLogger
from vec_env.subproc_vec_env import SubprocVecEnv
from sumo_ants_custom_env import *

## DAGGER imports
from openai_policy import LSTMPolicy, MlpPolicyValue
import pickle
import tensorflow as tf

def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params

def setFromFlat(var_list, flat_params):
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    tf.get_default_session().run(op, {theta: flat_params})

class Experiment():
    def __init__(self,
                 learner,
                 env_name,
                 tags = [],
                 states = {},
                 cuda = 0,
                 seed = None,
                 learner_kwargs=dict(),
                 logger_kwargs=dict(),
                 epochs=100,
                 steps_per_epoch=5000,
                 train_every_n_step = 500,
                 start_steps=10000,
                 replay_size=int(1e6),
                 save_freq=1,
                 rew_type = None,
                 freeze_opponent = False,
                 use_comet = False,
                 project_name = 'sumo_sac',
                 exp_name = "",
                 store_both = True,
                 agent_checkpoint = None,
                 decay_over_epochs = 500,
                 dagger_agent = ''
    ):
        if not seed : seed = np.random.randint(100000)
        if type(states) is str: states = states.split(',')
        if type(tags) is str: tags = tags.split(',')
        exp_name += "_" + str(uuid.uuid4())[:8]
        logger_kwargs['exp_name'] = exp_name
        logger_kwargs['use_comet'] = use_comet
        logger_kwargs['comet_args'] = {"project_name":project_name}
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())
        tags.append(learner.__name__)
        for tag in tags: self.logger.add_tag(tag)

        self.rew_type = rew_type
        self.states = states
        self.epochs = epochs
        self.train_every_n_step = train_every_n_step
        self.steps_per_epoch = steps_per_epoch
        self.start_steps = start_steps
        self.replay_size = replay_size
        self.save_freq = save_freq
        self.store_both = store_both
        self.n_workers = 4
        self.n_agents = 2
        self.decay_over_epochs = decay_over_epochs
        self.freeze_opponent = freeze_opponent

        self.device = torch.device(f"cuda:{cuda}"
                              if torch.cuda.is_available()
                              and cuda is not None else "cpu")

        self.test_env = make_env(env_name, seed)()

        # create env for each worker
        seed_shift = np.random.randint(1000)
        self.envs = [make_env(env_name, seed_shift + seed) for seed in range(self.n_workers)]
        self.envs = SubprocVecEnv(self.envs)

        self.obs_dim = self.test_env.observation_space.spaces[0].shape[0]
        self.act_dim = self.test_env.action_space.spaces[0].shape[0]
        self.act_limit = self.test_env.action_space.spaces[0].high[0]
        self.max_ep_len = self.test_env._max_episode_steps
        
        self.buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)
        
        # agent history storage prep
        self.most_recent_agent = None
        self.agents_buffer_path = 'data/' + exp_name + '/'
        os.makedirs(self.agents_buffer_path, exist_ok = True)
        
        learner_kwargs['device'] = self.device
        learner_kwargs['action_space'] = self.test_env.action_space.spaces[0]
        learner_kwargs['ac_kwargs'] = dict()
        learner_kwargs['ac_kwargs']['state_dim'] = self.obs_dim

        self.agent1 = learner(**learner_kwargs) # Agent that is trained
        self.agent2 = learner(**learner_kwargs) # Agent that is sampled from the past history opponents

        if agent_checkpoint:
            ckpt_agent = self.load_agent_checkpoint(agent_checkpoint)
            self.agent1.ac_main = copy.deepcopy(ckpt_agent.ac_main)
            self.agent1.ac_target = copy.deepcopy(ckpt_agent.ac_target)
            self.agent2.ac_main = copy.deepcopy(ckpt_agent.ac_main)
            self.agent2.ac_target = copy.deepcopy(ckpt_agent.ac_target)

        self.load_tf_agent(dagger_agent)
        self.set_seed(seed)

    def load_tf_agent(self, path):
        tf_config = tf.ConfigProto(
                    inter_op_parallelism_threads=1,
                    intra_op_parallelism_threads=1)
        self.sess = tf.Session(config=tf_config)
        self.sess.__enter__()
        scope = "policy" + str(0)
        self.expert = LSTMPolicy(scope=scope, reuse=False,
                                 ob_space=self.test_env.observation_space.spaces[0],
                                 ac_space=self.test_env.action_space.spaces[0],
                                 hiddens=[128, 128], normalize=True)
        self.sess.run(tf.variables_initializer(tf.global_variables()))
        params = load_from_file(param_pkl_path=path)
        setFromFlat(self.expert.get_variables(), params)
    
    def get_expert_action(self, obs):
        a = self.expert.act(stochastic=True, observation=obs)[0]
        return a
        
    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        # https://pytorch.org/docs/master/notes/randomness.html#cudnn
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def save(self,):
        self.logger.save_state({'env': self.test_env}, self.agent1.ac_main, None)

    def process_reward(self, rews, infos, epoch, obs):
        rewards_per_agent = []
        for agent_i, (env_rew, info) in enumerate(zip(rews, infos)):
            # https://github.com/openai/multiagent-competition/blob/master/gym-compete/gym_compete/new_envs/agents/ant_fighter.py
            # info[0] sample
            # { 'reward_center': -1.0324836626238088,
            #   'reward_ctrl': 0.0983036621497015,
            #   'reward_contact': 0.0,
            #   'reward_survive': 5.0,
            #   'reward_move': 3.86921267522649,
            #   'agent_done': False,
            #   'reward_remaining': 0.0},
            if self.rew_type == 'center':
                r_i = self.test_env.MAX_RADIUS + info['reward_center']
            elif self.rew_type == 'center3':
                r_i = np.power(self.test_env.MAX_RADIUS + info['reward_center'],3)
            elif self.rew_type == 'move':
                r_i = info['reward_move']
            elif self.rew_type == 'all':
                r_i = env_rew
            elif self.rew_type == 'all40':
                r_i = 40*env_rew
            elif self.rew_type == 'all3':
                if epoch < self.decay_over_epochs: # decay over n epochs, then 1.0 to env reward
                    alpha = epoch / self.decay_over_epochs
                    assert 0.0 <= alpha <= 1.0
                    center3_reward = np.power(self.test_env.MAX_RADIUS + info['reward_center'],3)
                    r_i = (1-alpha) * center3_reward + alpha * (20 * env_rew)
                else:
                    r_i = 20 * env_rew
            elif self.rew_type == 'opponent':
                a1_loc = obs[0][:2]
                a2_loc = obs[1][:2]
                opponent_distance = np.sqrt(np.sum((a1_loc - a2_loc)**2))
                max_arena_diameter = self.test_env.MAX_RADIUS * 2
                opponent_distance_reward = (max_arena_diameter - opponent_distance)**3
                r_i = opponent_distance_reward
            elif self.rew_type == 'opponent+envx20':
                a1_loc = obs[0][:2]
                a2_loc = obs[1][:2]
                opponent_distance = np.sqrt(np.sum((a1_loc - a2_loc)**2))
                max_arena_diameter = self.test_env.MAX_RADIUS * 2
                opponent_distance_reward = (max_arena_diameter - opponent_distance)**3
                r_i = opponent_distance_reward + env_rew * 20
            rewards_per_agent.append(r_i)
        return rewards_per_agent

    def load_agent_checkpoint(self, checkpoint_file):
        agent = torch.load(checkpoint_file)
        agent.device = self.device
        agent.ac_main.to(self.device)
        agent.ac_target.to(self.device)
        return agent

    def save_agent(self, agent):
        self.most_recent_agent = self.agents_buffer_path + f"{self.steps_count}.pt"
        torch.save(agent, self.most_recent_agent)

    def sample_agent(self,):
        # Uniformly at random sample one agent from the past
        random_agent = np.random.choice(glob.glob(f"{self.agents_buffer_path}*"))
        # Load the agent
        sample_agent = torch.load(random_agent)
        return sample_agent

    def improve(self, episode_len):
        self.agent1.train()
        # self.agent2.train()
        for _ in range(episode_len):
            # buffer or batch sample
            metrics = self.agent1.learn(self.buffer)
            self.logger.store(**metrics)
        # save agent 1
        self.save_agent(self.agent1)
        # sample agent 2
        self.agent2 = self.sample_agent()

    def evaluate(self, render = False, n = 10): #test - fully deterministic
        self.agent1.eval()
        for j in range(n):
            o, d, ep_ret_a1, ep_ret_a2, ep_len = self.test_env.reset(), (False, False), 0, 0, 0
            while not(d[0] or d[1]):
                self.test_env.render()
                # Take deterministic actions at test time
                a1 = self.agent1.get_action(o[0], deterministic = True)
                a2 = self.agent1.get_action(o[1], deterministic = True) # using best agent
                if self.freeze_opponent: a2 = np.zeros_like(a2)
                o, r, d, i = self.test_env.step((a1,a2))
                # r = self.process_reward(r, i, epoch = )
                ep_ret_a1 += r[0]
                ep_ret_a2 += r[1]
                ep_len += 1
            self.logger.store(TestEpRet_a1=ep_ret_a1, TestEpRet_a2=ep_ret_a2, TestEpLen=ep_len)

    def logger_checkpoint(self, epoch, start_time):
        self.logger.log_tabular('Epoch', epoch)
        learner_checkpoint_data = self.agent1.checkpoint()
        for key in ['EpRet_a1', 'EpRet_a2', 'TestEpRet_a1', 'TestEpRet_a2']:
            learner_checkpoint_data[key] = 'mm'
        for key in ['EpLen', 'TestEpLen']:
            learner_checkpoint_data[key] = 'avg'
        for key, flag in learner_checkpoint_data.items():
            if flag == 'mm':
                self.logger.log_tabular(key, with_min_and_max=True)
            elif flag == 'avg':
                self.logger.log_tabular(key, average_only=True)
            else:
                print('unknown checkpoint flag')
        # Log info about epoch
        self.logger.log_tabular('TotalEnvInteracts', self.steps_count)
        self.logger.log_tabular('Time', time.time()-start_time)
        self.logger.dump_tabular()

    def run_experiment(self): # main training loop
        start_time = time.time()
        total_steps = self.steps_per_epoch * self.epochs
        self.steps_count = 0
        # reset all of the envs
        o_ts = self.envs.reset()
        ep_rets = np.zeros((self.n_workers,self.n_agents), dtype=np.float32)
        ep_lens = np.zeros(self.n_workers, dtype=np.int32)
        # Main loop: collect experience in env and update/log each epoch
        while self.steps_count < total_steps:
            # Select action
            if self.steps_count > self.start_steps:
                obs_a1, obs_a2 = map(np.array, zip(*o_ts))
                a_a1 = self.agent1.get_action(obs_a1, deterministic = False).reshape(self.n_workers,1,-1)
                a_a2 = self.agent2.get_action(obs_a2, deterministic = False).reshape(self.n_workers,1,-1)
                actions = np.concatenate([a_a1, a_a2], axis=1)
            else: # DAGGER demos
                actions = np.zeros((self.n_workers, self.n_agents, self.act_dim))
                for w_i in range(self.n_workers):
                    for a_i in range(self.n_agents):
                        actions[w_i,a_i,:] = self.get_expert_action(o_ts[w_i,a_i,:])
            if self.freeze_opponent: actions[:,1,:] = np.zeros_like(actions[:,1,:])
            # Make an env step
            o2_ts, env_r_ts, d_ts, i_ts = self.envs.step(actions)
            r_ts = np.zeros((self.n_workers,self.n_agents))
            envs_were_reset = []
            train_time = False
            evaluation_time = False
            epoch = self.steps_count // self.steps_per_epoch
            for i, (env, env_r_t, d_t, i_t, o2_t) in enumerate(zip(self.envs.remotes, env_r_ts, d_ts, i_ts, o2_ts)):
                if d_t[0] or d_t[1]: envs_were_reset.append(i)
                # Ignore the "done" signal if it comes from hitting the time
                # horizon (that is, when it's an artificial terminal signal
                # that isn't based on the agent's state)
                d_ts[i] = [False, False] if ep_lens[i]==self.max_ep_len else d_ts[i]

                r_t = self.process_reward(env_r_t, i_t, epoch, o2_t)
                r_ts[i,:] = r_t
                ep_rets[i,:] += r_t
                ep_lens[i] += 1

                self.steps_count += 1
                if self.steps_count % self.steps_per_epoch == 0: evaluation_time = True
                if self.steps_count % self.train_every_n_step == 0: train_time = True
            # Store buffer - only agent1 experiences
            if self.store_both:
                self.buffer.store_batch(o_ts, actions, r_ts, o2_ts, d_ts) # store both agent's experiences 
            else:
                self.buffer.store_batch(*[np.delete(item,1,1) for item in [o_ts, actions, r_ts, o2_ts, d_ts]])

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o_ts = o2_ts

            for i in envs_were_reset:
                if d_ts[i][0] or d_ts[i][1]:
                    self.logger.store(EpRet_a1=ep_rets[i][0], EpRet_a2=ep_rets[i][1], EpLen=ep_lens[i])
                    o_ts[i] = self.envs.reset_env(i)
                    ep_rets[i, :] = np.array([0,0])
                    ep_lens[i] = 0

            # Train - update the learner
            if train_time: self.improve(self.train_every_n_step)

            # Evaluate
            if evaluation_time:
                # Test the performance of the deterministic version of the best agent.
                self.evaluate()
                self.logger_checkpoint(epoch, start_time)
                # Save model
                if (epoch % self.save_freq == 0) or (epoch == self.epochs-1):
                    self.save()
                    self.logger.upload_file(self.most_recent_agent) # upload agent to comet.ml
        self.wrapup()

    def wrapup(self,):
        self.envs.close()

def make_env(env_id, seed=None):
    def _f():
        env = gym.make(env_id)
#         env.move_reward_weight = 0.0 # TODO check if necessary
        print('seed: ',seed)
        if seed is not None: env.seed(seed)
        return env
    return _f

if __name__ == "__main__":
    #TODO argparser
    from learners.sac import SAC
    CUDA = 0
    learner_kwargs = dict(
        batch_size = 128,
        polyak = 0.995,
    )

    dagger_agent_path = 'agent-zoo/sumo/ants/agent_parameters-v2.pkl'
    exp = Experiment(
        SAC,
        env_name = 'sumo-ants-v0',
        save_freq=10,
        epochs = 3000,
        start_steps=int(1e6),
#         start_steps=200000,
        cuda = CUDA,
        learner_kwargs = learner_kwargs,
        use_comet = True,
        freeze_opponent = False,
        # agent_checkpoint = 'data/center3_sac_842dce98/3204500.pt', # best walker to the center (center^3)
        agent_checkpoint = 'data/opponent_sac_18f4ccdb/1903000.pt', # best to opponent reward agent
#         agent_checkpoint = 'data/dagger_sac_600k_0move_48cd484d/1212500.pt', # best after dagger        rew_type = 'all',
#         rew_type = 'opponent',
#         rew_type = 'opponent+envx20',
        exp_name = 'dagger_sac_1M_original',
        tags = 'scratch',
        dagger_agent = dagger_agent_path
    )
    exp.run_experiment()
