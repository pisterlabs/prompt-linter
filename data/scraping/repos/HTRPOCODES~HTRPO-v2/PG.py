import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np
import basenets
from agents.Agent import Agent
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import copy
from .config import PG_CONFIG
from rlnets import FCPG_Gaussian, FCPG_Softmax, FCVALUE, ConvPG_Softmax
from utils import databuffer, databuffer_PG_gaussian, databuffer_PG_softmax
import abc
import os
from collections import deque
from utils.mathutils import explained_variance

class PG(Agent):
    __metaclass__ = abc.ABCMeta
    def __init__(self,hyperparams):
        config = copy.deepcopy(PG_CONFIG)
        config.update(hyperparams)
        super(PG, self).__init__(config)
        self.max_grad_norm = config['max_grad_norm']
        self.nsteps = config['steps_per_iter']
        self.optimizer_type = config['optimizer']
        self.entropy_weight = config['entropy_weight']
        self.value_type = config['value_type']
        self.policy_type = config['policy_type']
        self.using_KL_estimation = config['using_KL_estimation']
        self.policy_ent = 0.
        self.policy_loss = 0.
        self.value_loss = 0.

        # initialize value approximator
        if self.value_type is not None:
            # initialize value network architecture
            if self.value_type == 'FC':
                self.value = FCVALUE(
                    self.n_states['n_s'] + self.n_states['n_g'] if isinstance(self.n_states, dict) else self.n_states,
                    n_hiddens=config['hidden_layers_v']
                    if isinstance(config['hidden_layers_v'], list)
                    else config['hidden_layers'],
                    usebn=self.using_bn,
                    nonlinear=self.act_func,
                    )
            # choose estimator, including Q, A and GAE.
            self.lamb = config['GAE_lambda']
            # value approximator optimizer
            self.v_loss_reduction = config['v_loss_reduction'] if 'v_loss_reduction' in config else None
            self.loss_func_v = config['loss_func_v'](reduction = self.v_loss_reduction if self.v_loss_reduction else 'mean')
            self.lr_v = config['lr_v']
            self.mom_v = config['mom_v']
            self.iters_v = config['iters_v']
            if config['v_optimizer'] == optim.LBFGS:
                self.using_lbfgs_for_V = True
            else:
                self.using_lbfgs_for_V = False
                if self.mom is not None:
                    self.v_optimizer = config['v_optimizer'](self.value.parameters(), lr=self.lr_v, momentum = self.mom_v)
                else:
                    self.v_optimizer = config['v_optimizer'](self.value.parameters(), lr=self.lr_v)
        elif self.value_type is None:
            self.value = None

    def cuda(self):
        Agent.cuda(self)
        self.policy = self.policy.cuda()
        if self.value_type is not None:
            self.value = self.value.cuda()

    @abc.abstractmethod
    def _random_action(self, n):
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def sample_batch(self, batchsize = None):
        raise NotImplementedError("Must be implemented in subclass.")

    # compute importance sampling factor between current policy and previous trajectories
    @abc.abstractmethod
    def compute_imp_fac(self, inds = None, model= None):
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def compute_entropy(self, inds = None, model= None):
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def mean_kl_divergence(self, inds = None, model= None):
        raise NotImplementedError("Must be implemented in subclass.")

    def estimate_value_with_approximator(self):
        fake_done = torch.nonzero(self.done.squeeze() == 2).squeeze(-1)
        self.done[self.done == 2] = 1
        returns = torch.zeros(self.r.size(0), 1).type_as(self.s)
        values = self.value(self.s, other_data=self.other_data)
        delta = torch.zeros(self.r.size(0), 1).type_as(self.s)
        advantages = torch.zeros(self.r.size(0), 1).type_as(self.s)

        # mask is a 1-d vector, therefore, e_points is also a 1-d vector
        e_points = torch.nonzero(self.done.squeeze() == 1).squeeze()
        b_points = - torch.ones(size=e_points.size()).type_as(e_points)
        b_points[1:] = e_points[:-1]
        ep_lens = e_points - b_points
        assert ep_lens.min().item() > 0, "Some episode lengths are smaller than 0."
        max_len = torch.max(ep_lens).item()
        uncomplete_flag = ep_lens > 0

        delta[e_points] = self.r[e_points] - values[e_points]
        if fake_done.numel() > 0:
            delta[fake_done] += self.value(self.s_[fake_done],
                                           other_data=self.other_data[fake_done]
                                           if self.other_data is not None else None).resize_as(delta[fake_done])
        advantages[e_points] = delta[e_points]
        returns[e_points] = self.r[e_points]
        for i in range(1, max_len):
            uncomplete_flag[ep_lens <= i] = 0
            # TD-error
            inds = (e_points - i)[uncomplete_flag]
            delta[inds] = self.r[inds] + self.gamma * values[inds + 1] - values[inds]
            advantages[inds] = delta[inds] + self.gamma * self.lamb * advantages[inds + 1]
            returns[inds] = self.r[inds] + self.gamma * returns[inds + 1]

        # Estimated Return, from OpenAI baseline.
        esti_return = values + advantages
        # values returns advantages and estimated returns
        self.V = values.squeeze().detach()
        self.R = returns.squeeze().detach()
        self.A = advantages.squeeze().detach()
        self.esti_R = esti_return.squeeze().detach()

    def estimate_value_with_mc(self):
        self.done[self.done == 2] = 1
        returns = torch.zeros(self.r.size(0), 1).type_as(self.s)
        e_points = torch.nonzero(self.done.squeeze() == 1).squeeze()
        b_points = - torch.ones(size=e_points.size()).type_as(e_points)
        b_points[1:] = e_points[:-1]
        ep_lens = e_points - b_points
        assert ep_lens.min().item() > 0, "Some episode lengths are smaller than 0."
        max_len = torch.max(ep_lens).item()
        uncomplete_flag = ep_lens > 0
        returns[e_points] = self.r[e_points]
        for i in range(1, max_len):
            uncomplete_flag[ep_lens <= i] = 0
            inds = (e_points - i)[uncomplete_flag]
            returns[inds] = self.r[inds] + self.gamma * returns[inds + 1]

        self.R = returns.squeeze().detach()
        # Here self.A is actually not advantages. It works for policy updates, which
        # means that it is used to measure how good a action is.
        self.A = self.R

    def estimate_value(self):
        if self.value_type is not None:
            self.estimate_value_with_approximator()
        else:
            self.estimate_value_with_mc()

    def optim_value_lbfgs(self,V_target, inds):
        value = self.value
        value.zero_grad()
        loss_fn = self.loss_func_v
        def V_closure():
            predicted = value(self.s[inds], other_data = self.other_data[inds] if self.other_data is not None else None).squeeze()
            loss = loss_fn(predicted, V_target)
            self.value_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            return loss
        old_params = parameters_to_vector(value.parameters())
        for lr in self.lr * .5 ** np.arange(10):
            optimizer = optim.LBFGS(self.value.parameters(), lr=lr)
            optimizer.step(V_closure)
            current_params = parameters_to_vector(value.parameters())
            if any(np.isnan(current_params.data.cpu().numpy())):
                print("LBFGS optimization diverged. Rolling back update...")
                vector_to_parameters(old_params, value.parameters())
            else:
                return

    def update_value(self, inds = None):
        if inds is None:
            inds = np.arange(self.s.size(0))
        V_target = self.esti_R[inds]
        if self.using_lbfgs_for_V:
            self.optim_value_lbfgs(V_target, inds)
        else:
            V_eval = self.value(self.s[inds], other_data = self.other_data[inds] if self.other_data is not None else None).squeeze()
            self.loss_v = self.loss_func_v(V_eval, V_target)
            self.value_loss = self.loss_v.item()
            self.value.zero_grad()
            self.loss_v.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
            self.v_optimizer.step()

    def learn(self):
        self.sample_batch()
        self.estimate_value()
        imp_fac = self.compute_imp_fac()
        # update policy
        self.A = (self.A - self.A.mean()) / (self.A.std() + 1e-10)
        self.loss = - (imp_fac * self.A.squeeze()).mean()
        if self.value_type is not None:
            # update value
            for i in range(self.iters_v):
                self.update_value()
        self.policy.zero_grad()
        self.loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.learn_step_counter += 1
        self.cur_kl = self.mean_kl_divergence().item()
        self.policy_loss = self.loss.item()
        self.policy_ent = self.compute_entropy().item()

    def save_model(self, save_path):
        print("saving models...")
        save_dict = {
            'model': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'episode': self.episode_counter,
            'step': self.learn_step_counter,
        }
        torch.save(save_dict, os.path.join(save_path, "policy" + str(self.learn_step_counter) + ".pth"))
        if self.value_type is not None and not self.using_lbfgs_for_V:
            save_dict = {
                'model': self.value.state_dict(),
                'optimizer': self.v_optimizer.state_dict(),
            }
            torch.save(save_dict, os.path.join(save_path, "value" + str(self.learn_step_counter) + ".pth"))

    def load_model(self, load_path, load_point):
        policy_name = os.path.join(load_path, "policy" + str(load_point) + ".pth")
        print("loading checkpoint %s" % (policy_name))
        checkpoint = torch.load(policy_name)
        self.policy.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.learn_step_counter = checkpoint['step']
        self.episode_counter = checkpoint['episode']
        print("loaded checkpoint %s" % (policy_name))

        if self.value_type is not None and not self.using_lbfgs_for_V:
            value_name = os.path.join(load_path, "value" + str(load_point) + ".pth")
            print("loading checkpoint %s" % (value_name))
            checkpoint = torch.load(value_name)
            self.value.load_state_dict(checkpoint['model'])
            self.v_optimizer.load_state_dict(checkpoint['optimizer'])
            print("loaded checkpoint %s" % (value_name))


class PG_Gaussian(PG):
    def __init__(self,hyperparams):
        config = copy.deepcopy(PG_CONFIG)
        config.update(hyperparams)
        super(PG_Gaussian, self).__init__(config)

        # initialize buffer
        if "memory_size" not in config.keys():
            config['memory_size'] = self.memory_size
        self.memory = databuffer_PG_gaussian(config)

        # initialize policy net
        self.action_bounds = config['action_bounds']
        self.init_noise = config['init_noise']
        if self.policy_type == 'FC':
            self.policy = FCPG_Gaussian(
                self.n_states['n_s'] + self.n_states['n_g'] if isinstance(self.n_states, dict) else self.n_states,
                # input dim
                self.n_action_dims,  # output dim
                sigma=self.init_noise,
                outactive=self.out_act_func,
                outscaler=self.action_bounds,
                n_hiddens=self.hidden_layers,
                nonlinear=self.act_func,
                usebn=self.using_bn)
        elif self.policy_type == 'Conv':
            raise NotImplementedError

        # initialize optimizer
        if self.mom is not None:
            self.optimizer = self.optimizer_type(self.policy.parameters(), lr=self.lr, momontum = self.mom)
        else:
            self.optimizer = self.optimizer_type(self.policy.parameters(), lr=self.lr)

        # initialize data
        self.mu = torch.Tensor(1)
        self.sigma = torch.Tensor(1)

    def cuda(self):
        PG.cuda(self)
        self.mu = self.mu.cuda()
        self.sigma = self.sigma.cuda()

    def _random_action(self, n):
        random_a = np.random.uniform(low=-self.action_bounds, high=self.action_bounds, size=(n, self.n_action_dims))
        return torch.Tensor(random_a).type_as(self.a)

    def choose_action(self, s, other_data = None, greedy = False):
        self.policy.eval()
        if self.use_cuda:
            s = s.cuda()
            if other_data is not None:
                other_data = other_data.cuda()
        mu, logsigma, sigma = self.policy(s, other_data)
        mu = mu.detach()
        logsigma = logsigma.detach()
        sigma = sigma.detach()
        self.policy.train()
        if not greedy:
            a = torch.normal(mu,sigma)
        else:
            a = mu
        return a, mu, logsigma, sigma

    def compute_logp(self,mu,logsigma, sigma,a):
        if a.dim() == 1:
            return -0.5 * torch.sum(torch.pow((a - mu) / sigma, 2)) \
                    -0.5 * self.n_action_dims * torch.log(torch.Tensor([2. * np.pi]).type_as(mu)) \
                    -torch.sum(logsigma)
        elif a.dim() == 2:
            return  -0.5 * torch.sum(torch.pow((a - mu) / sigma, 2), 1) \
                    -0.5 * self.n_action_dims * torch.log(torch.Tensor([2. * np.pi]).type_as(mu)) \
                    -torch.sum(logsigma, 1)
        else:
            RuntimeError("a must be a 1-D or 2-D Tensor or Variable")

    def compute_imp_fac(self, inds = None, model = None):
        # default: compute all importance factors.
        if inds is None:
            inds = np.arange(self.s.size(0))
        if model is None:
            model = self.policy
        # theta is the vectorized model parameters
        mu_now, logsigma_now, sigma_now = model(self.s[inds], other_data = self.other_data[inds] if self.other_data is not None else None)
        # important sampling coefficients
        # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
        imp_fac = torch.exp(
            self.compute_logp(mu_now, logsigma_now, sigma_now, self.a[inds]) - self.logpac_old[inds].squeeze())
        return imp_fac

    def compute_entropy(self, inds = None, model = None):
        # default: compute all importance factors.
        if inds is None:
            inds = np.arange(self.s.size(0))
        if model is None:
            model = self.policy
        mu_now, logsigma_now, _ = model(self.s[inds], other_data = self.other_data[inds] if self.other_data is not None else None)
        entropy = (0.5 * self.n_action_dims * np.log(2 * np.pi * np.e) + torch.sum(logsigma_now, 1)).mean()
        return entropy

    def mean_kl_divergence(self, inds = None, model = None):
        if inds is None:
            inds = np.arange(self.s.size(0))
        if model is None:
            model = self.policy
        mu1, logsigma1, sigma1 = model(self.s[inds], other_data = self.other_data[inds] if self.other_data is not None else None)
        if self.using_KL_estimation:
            logp = self.compute_logp(mu1, logsigma1, sigma1, self.a[inds])
            logp_old = self.logpac_old[inds].squeeze()
            kl = 0.5 * torch.mean(torch.pow((logp_old - logp),2))
        else:
            mu2, logsigma2, sigma2 = self.mu[inds], torch.log(self.sigma[inds]), self.sigma[inds]
            sigma1 = torch.pow(sigma1, 2)
            sigma2 = torch.pow(sigma2, 2)
            kl = 0.5 * (torch.sum(torch.log(sigma1) - torch.log(sigma2), dim=1) - self.n_action_dims +
                        torch.sum(sigma2 / sigma1, dim=1) + torch.sum(torch.pow((mu1 - mu2), 2) / sigma1, 1)).mean()
        return kl

    def sample_batch(self, batch_size = None):
        batch, self.sample_index = Agent.sample_batch(self)
        self.r = self.r.resize_(batch['reward'].shape).copy_(torch.Tensor(batch['reward']))
        self.done = self.done.resize_(batch['done'].shape).copy_(torch.Tensor(batch['done']))
        self.a = self.a.resize_(batch['action'].shape).copy_(torch.Tensor(batch['action']))
        self.s = self.s.resize_(batch['state'].shape).copy_(torch.Tensor(batch['state']))
        self.s_ = self.s_.resize_(batch['next_state'].shape).copy_(torch.Tensor(batch['next_state']))
        self.logpac_old = self.logpac_old.resize_(batch['logpac'].shape).copy_(torch.Tensor(batch['logpac']))
        self.mu = self.mu.resize_(batch['mu'].shape).copy_(torch.Tensor(batch['mu']))
        self.sigma = self.sigma.resize_(batch['sigma'].shape).copy_(torch.Tensor(batch['sigma']))
        self.other_data = batch['other_data']
        if self.other_data:
            for key in self.other_data.keys():
                self.other_data[key] = torch.Tensor(self.other_data[key]).type_as(self.s)

class PG_Softmax(PG):
    def __init__(self,hyperparams):
        config = copy.deepcopy(PG_CONFIG)
        config.update(hyperparams)
        super(PG_Softmax, self).__init__(config)

        # initialize buffer
        config['memory_size'] = self.memory_size
        self.memory = databuffer_PG_softmax(config)

        # initialize policy net
        if self.policy_type == 'FC':
            self.policy = FCPG_Softmax(
                self.n_states['n_s'] + self.n_states['n_g'] if isinstance(self.n_states, dict) else self.n_states,
                # input dim
                self.n_actions,  # output dim
                n_hiddens=self.hidden_layers,
                nonlinear=self.act_func,
                usebn=self.using_bn,
                )
        elif self.policy_type == 'Conv':
            self.policy = ConvPG_Softmax(
                self.n_states,
                # input dim
                self.n_actions,  # output dim
                fcs=self.hidden_layers,
                nonlinear=self.act_func,
                usebn=self.using_bn,
            )

        # initialize optimizer
        if self.mom is not None:
            self.optimizer = self.optimizer_type(self.policy.parameters(), lr=self.lr, momontum = self.mom)
        else:
            self.optimizer = self.optimizer_type(self.policy.parameters(), lr=self.lr)

        # initialize data
        self.distri = torch.Tensor(1)

    def cuda(self):
        PG.cuda(self)
        self.distri = self.distri.cuda()

    def _random_action(self, n):
        return torch.multinomial(1. / self.n_actions * torch.ones(self.n_actions), n, replacement = True).type_as(self.a)

    def choose_action(self, s, other_data = None, greedy = False):
        self.policy.eval()
        if self.use_cuda:
            s = s.cuda()
            if other_data is not None:
                other_data = other_data.cuda()
        distri = self.policy(s, other_data).detach()
        self.policy.train()
        if not greedy:
            a = torch.multinomial(distri, 1, replacement=True)
        else:
            _, a = torch.max(distri, dim=-1, keepdim=True)
        # a = np.random.choice(distri.shape[0], p = distri.cpu().numpy())
        return a, distri

    def compute_logp(self,distri,a):
        if distri.dim() == 1:
            return torch.log(distri[a] + 1e-10)
        elif distri.dim() == 2:
            a_indices = [range(distri.size(0)),a.squeeze(-1).long().cpu().numpy().tolist()]
            return torch.log(distri[a_indices] + 1e-10)
        else:
            RuntimeError("distri must be a 1-D or 2-D Tensor or Variable")

    def compute_imp_fac(self, inds = None, model = None):
        # default: compute all importance factors.
        if inds is None:
            inds = np.arange(self.s.size(0))
        if model is None:
            model = self.policy
        distri_now = model(self.s[inds], other_data = self.other_data[inds] if self.other_data is not None else None)
        # important sampling coefficients
        # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
        imp_fac = torch.exp(self.compute_logp(distri_now, self.a[inds]) - self.logpac_old[inds].squeeze())
        return imp_fac

    def compute_entropy(self, inds = None, model = None):
        # default: compute all importance factors.
        if inds is None:
            inds = np.arange(self.s.size(0))
        if model is None:
            model = self.policy
        distri = model(self.s[inds], other_data = self.other_data[inds] if self.other_data is not None else None)
        # important sampling coefficients
        # imp_fac: should be a 1-D Variable or Tensor, size is the same with a.size(0)
        entropy = - torch.sum(distri * torch.log(distri), 1).mean()
        return entropy

    def mean_kl_divergence(self, inds = None, model = None):
        if inds is None:
            inds = np.arange(self.s.size(0))
        if model is None:
            model = self.policy
        distri1 = model(self.s[inds], other_data = self.other_data[inds] if self.other_data is not None else None)
        distri2 = self.distri[inds].squeeze()
        logratio = torch.log(distri2 / distri1)
        kl = torch.sum(distri2 * logratio, 1)
        return kl.mean()

    def sample_batch(self, batch_size = None):
        batch, self.sample_index = Agent.sample_batch(self)
        self.r = self.r.resize_(batch['reward'].shape).copy_(torch.Tensor(batch['reward']))
        self.done = self.done.resize_(batch['done'].shape).copy_(torch.Tensor(batch['done']))
        self.a = self.a.resize_(batch['action'].shape).copy_(torch.Tensor(batch['action']))
        self.s = self.s.resize_(batch['state'].shape).copy_(torch.Tensor(batch['state']))
        self.s_ = self.s_.resize_(batch['next_state'].shape).copy_(torch.Tensor(batch['next_state']))
        self.logpac_old = self.logpac_old.resize_(batch['logpac'].shape).copy_(torch.Tensor(batch['logpac']))
        self.distri = self.distri.resize_(batch['distri'].shape).copy_(torch.Tensor(batch['distri']))
        self.other_data = batch['other_data']
        if self.other_data:
            for key in self.other_data.keys():
                self.other_data[key] = torch.Tensor(self.other_data[key]).type_as(self.s)

def run_pg_train(env, agent, max_timesteps, logger):
    timestep_counter = 0
    total_updates = max_timesteps // agent.nsteps
    epinfobuf = deque(maxlen=100)

    while(True):
        mb_obs, mb_rewards, mb_actions, mb_dones, mb_logpacs, mb_obs_, mb_mus, mb_sigmas \
            , mb_distris= [], [], [], [], [], [], [], [], []
        epinfos = []
        observations = env.reset()
        for i in range(0, agent.nsteps, env.num_envs):
            observations = torch.Tensor(observations)
            if not agent.dicrete_action:
                actions, mus, logsigmas, sigmas = agent.choose_action(observations)
                logp = agent.compute_logp(mus, logsigmas, sigmas, actions)
                mus = mus.cpu().numpy()
                sigmas = sigmas.cpu().numpy()
                mb_mus.append(mus)
                mb_sigmas.append(sigmas)
            else:
                actions, distris = agent.choose_action(observations)
                logp = agent.compute_logp(distris, actions)
                distris = distris.cpu().numpy()
                mb_distris.append(distris)
            observations = observations.cpu().numpy()
            actions  = actions.cpu().numpy()
            logp = logp.cpu().numpy()
            observations_, rewards, dones, infos = env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_obs.append(observations)
            mb_actions.append(actions)
            mb_logpacs.append(logp)
            mb_dones.append(dones.astype(np.uint8))
            mb_rewards.append(rewards)
            mb_obs_.append(observations_)
            observations = observations_

        epinfobuf.extend(epinfos)
        # make all final states marked by done, preventing wrong estimating of returns and advantages.
        # done flag:
        #      0: undone and not the final state
        #      1: realdone
        #      2: undone but the final state
        mb_dones[-1][np.where(mb_dones[-1] == 0)] = 2

        def reshape_data(arr):
            s = arr.shape
            return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
        mb_obs = reshape_data(np.asarray(mb_obs, dtype=np.float32))
        mb_rewards = reshape_data(np.asarray(mb_rewards, dtype=np.float32))
        mb_actions = reshape_data(np.asarray(mb_actions))
        mb_logpacs = reshape_data(np.asarray(mb_logpacs, dtype=np.float32))
        mb_dones = reshape_data(np.asarray(mb_dones, dtype=np.uint8))
        mb_obs_ = reshape_data(np.asarray(mb_obs_, dtype=np.float32))

        assert mb_obs.ndim <= 2 and mb_rewards.ndim <= 2 and mb_actions.ndim <= 2 and \
               mb_logpacs.ndim <= 2 and mb_dones.ndim <= 2 and mb_obs_.ndim <= 2, \
            "databuffer only supports 1-D data's batch."

        if not agent.dicrete_action:
            mb_mus = reshape_data(np.asarray(mb_mus, dtype=np.float32))
            mb_sigmas = reshape_data(np.asarray(mb_sigmas, dtype=np.float32))
            assert mb_mus.ndim <= 2 and mb_sigmas.ndim <= 2, "databuffer only supports 1-D data's batch."
        else:
            mb_distris = reshape_data(np.asarray(mb_distris, dtype=np.float32))
            assert mb_distris.ndim <= 2, "databuffer only supports 1-D data's batch."

        # store transition
        transition = {
            'state': mb_obs if mb_obs.ndim == 2 else np.expand_dims(mb_obs, 1),
            'action': mb_actions if mb_actions.ndim == 2 else np.expand_dims(mb_actions, 1),
            'reward': mb_rewards if mb_rewards.ndim == 2 else np.expand_dims(mb_rewards, 1),
            'next_state': mb_obs_ if mb_obs_.ndim == 2 else np.expand_dims(mb_obs_, 1),
            'done': mb_dones if mb_dones.ndim == 2 else np.expand_dims(mb_dones, 1),
            'logpac': mb_logpacs if mb_logpacs.ndim == 2 else np.expand_dims(mb_logpacs, 1),
        }
        if not agent.dicrete_action:
            transition['mu'] = mb_mus if mb_mus.ndim == 2 else np.expand_dims(mb_mus, 1)
            transition['sigma'] = mb_sigmas if mb_sigmas.ndim == 2 else np.expand_dims(mb_sigmas, 1)
        else:
            transition['distri'] = mb_distris if mb_distris.ndim == 2 else np.expand_dims(mb_distris, 1)
        agent.store_transition(transition)

        # agent learning step
        agent.learn()

        # training controller
        timestep_counter += agent.nsteps
        if timestep_counter >= max_timesteps:
            break

        # adjust learning rate for policy and value function
        decay_coef = 1 - agent.learn_step_counter / total_updates
        adjust_learning_rate(agent.optimizer, original_lr=agent.lr, decay_coef=decay_coef)
        if agent.value_type is not None:
            adjust_learning_rate(agent.v_optimizer, original_lr=agent.lr_v, decay_coef=decay_coef)

        print("------------------log information------------------")
        print("total_timesteps:".ljust(20) + str(timestep_counter))
        print("iterations:".ljust(20) + str(agent.learn_step_counter) + " / " + str(int(total_updates)))
        if agent.value_type is not None:
            explained_var = explained_variance(agent.V.cpu().numpy(), agent.esti_R.cpu().numpy())
            print("explained_var:".ljust(20) + str(explained_var))
            logger.add_scalar("explained_var/train", explained_var, timestep_counter)
        print("episode_len:".ljust(20) + "{:.1f}".format(np.mean([epinfo['l'] for epinfo in epinfobuf])))
        print("episode_rew:".ljust(20) + str(np.mean([epinfo['r'] for epinfo in epinfobuf])))
        logger.add_scalar("episode_reward/train", np.mean([epinfo['r'] for epinfo in epinfobuf]), timestep_counter)
        print("mean_kl:".ljust(20) + str(agent.cur_kl))
        logger.add_scalar("mean_kl/train", agent.cur_kl, timestep_counter)
        print("policy_ent:".ljust(20) + str(agent.policy_ent))
        logger.add_scalar("policy_ent/train", agent.policy_ent, timestep_counter)
        print("policy_loss:".ljust(20)+ str(agent.policy_loss))
        logger.add_scalar("policy_loss/train", agent.policy_loss, timestep_counter)
        print("value_loss:".ljust(20)+ str(agent.value_loss))
        logger.add_scalar("value_loss/train", agent.value_loss, timestep_counter)
    return agent

def adjust_learning_rate(optimizer, original_lr = 1e-4, decay_coef = 0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] = original_lr * decay_coef
