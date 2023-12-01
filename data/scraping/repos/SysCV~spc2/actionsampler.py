import os
import copy
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from utils.util import norm_image


class ActionSampleManager:
    def __init__(self, args, guides):
        self.args = args
        self.prev_act = np.array([1.0, 0.0])
        self.guides = guides
        self.cand_num = 20
        self.top_k = 5
        self.p = None
        self.pstep = self.args.pred_step
        self.time_discount = 0.5**torch.range(0, self.pstep-1)
        self.time_discount = torch.clamp(self.time_discount, 1/8., 1.)
        if torch.cuda.is_available():
            self.time_discount = self.time_discount.cuda()

    def get_guide_action(self, action, lb=-1.0, ub=1.0):
        # get the index of target bin in guidance grid
        _bin_divide = np.array(self.args.bin_divide)
        action = ((action - lb) / (ub - lb) * _bin_divide).astype(np.uint8)
        dim = len(self.args.bin_divide)
        weight = list(map(lambda x: np.prod(_bin_divide[:x]), range(dim)))
        weight = np.array(weight)
        return np.sum(action * weight, axis=-1)
  
    def generate_episode(self, mean, lb=-1.0, ub=1.0):
        res = []
        semi_range = (ub - lb) / 2.0
        uni_sample = lambda x: np.random.uniform(low=-semi_range/x, high=semi_range/x)
        for i in range(self.args.pred_step):
            rand = list(map(uni_sample, self.args.bin_divide))
            res.append(np.array(mean + rand))
        res = list(map(lambda x: x.reshape(1, -1), res))
        return np.concatenate(res, axis=0)

    def generate_action(self, p, size, guides, lb=-1.0, ub=1.0):
        res = []
        for _ in range(size):
            c = np.random.choice(range(len(p)), p=p)
            res.append(np.expand_dims(self.generate_episode(guides[c], lb, ub), axis=0))
        return np.concatenate(res, axis=0)

    def add_cost(self, output, key, speeds, weight, with_cur=False):
        pred = output[key]
        # with_cur indicates whether the prediction contains that on the current frame
        if with_cur:
            pred = pred[:, 1:]
        pred = F.softmax(pred, -1) # predicts binary events
        
        # calculate cost value for one cost item
        to_round = (self.args.sample_type == 'binary')
        if key == "colls_with_prob":
            pred_pos = torch.round(pred[:, :, :, 0]) if to_round else pred[:, :, :, 0] 
            pred_neg = torch.round(pred[:, :, :, 1]) if to_round else pred[:, :, :, 1]
            anchor_num = pred.shape[2]
            speeds = speeds.unsqueeze(-1).repeat([1, 1, anchor_num])
        else:
            pred_pos = torch.round(pred[:, :, 0]) if to_round else pred[:, :, 0] 
            pred_neg = torch.round(pred[:, :, 1]) if to_round else pred[:, :, 1]

        cost = -pred_pos * speeds + pred_neg * self.args.speed_threshold
        cost = cost * self.time_discount
        if key == "colls_with_prob":
            cost = cost.sum(axis=2)
        cost = (cost.view(-1, self.args.pred_step, 1) * weight).sum(-1).sum(-1)
        return cost

    def estimate_cost(self, net, imgs, actions, action_var=None, hidden=None, cell=None):
        batch_size = int(imgs.size()[0])

        weight = (self.args.time_decay ** np.arange(self.args.pred_step)).reshape((1, self.args.pred_step, 1))
        weight = Variable(torch.from_numpy(weight).float().cuda()).repeat(batch_size, 1, 1)

        output = net(imgs, actions, hidden=hidden, cell=cell, training=False, action_var=action_var)

        cost = 0
        speeds = output['speed'].view(-1, self.args.pred_step)

        use_coll = (self.args.sample_with_collision and self.args.use_collision)
        use_ins_coll = (use_coll and self.args.use_colls_with)
        use_offroad = (self.args.sample_with_offroad and self.args.use_offroad)
        use_offlane = (self.args.sample_with_offlane and self.args.use_offlane)

        if use_coll: cost += self.add_cost(output, 'coll_prob', speeds, weight)
        # if use_ins_coll: cost += self.add_cost(output, 'colls_with_prob', speeds, weight, with_cur=True)
        if use_offroad: cost += self.add_cost(output, 'offroad_prob', speeds, weight)
        if use_offlane: cost += self.add_cost(output, 'offlane_prob', speeds, weight)
        if use_ins_coll and self.args.SAS: 
            ins_cos = self.add_cost(output, 'colls_with_prob', speeds, weight, with_cur=True)
        else:
            ins_cos = 0

        return cost, ins_cos

    def _sample_action(self, p, net, imgs, guides, action_var=None, testing=False):
        imgs = copy.deepcopy(imgs)
        imgs = norm_image(imgs)

        batch_size, c, w, h = int(imgs.size()[0]), int(imgs.size()[-3]), int(imgs.size()[-2]), int(imgs.size()[-1])
        imgs = imgs.view(batch_size, 1, c, w, h)

        imgs = imgs.repeat(self.cand_num, 1, 1, 1, 1)
        action_var = action_var.repeat(self.cand_num, 1, 1)
        
        # generate action candidates from guidances
        action = self.generate_action(p, self.cand_num, guides)

        this_action0 = copy.deepcopy(action)
        this_action = Variable(torch.from_numpy(action).cuda().float(), requires_grad=False)
        with torch.no_grad():
            cost, ins_cost = self.estimate_cost(net, imgs, this_action, action_var, None, None).data.cpu().numpy()
        
        idx = np.argpartition(cost, self.top_k)
        top_k_idx = idx[:self.top_k]
        top_k_ins_cost = ins_cost[top_k_idx]
        idx = np.argmin(top_k_ins_cost)
        true_idx = top_k_idx[idx]
        res = this_action0[true_idx, :, :]
        
        if not testing:
            return res[0]
        else:
            return res

    def sample_action(self, net, obs, obs_var, action_var, exploration, step, explore=False, testing=False):
        if random.random() <= 1 - exploration.value(step) or not explore:
            obs = torch.from_numpy(np.expand_dims(obs.transpose(2, 0, 1), axis=0).copy()).float()
            obs = norm_image(obs)

            if torch.cuda.is_available():
                obs = obs.cuda()
            with torch.no_grad():
                obs = obs.repeat(max(1, torch.cuda.device_count()), 1, 1, 1)
                self.p = net(obs, action_only=True)[0]
                p = F.softmax(self.p / self.args.temperature, dim=-1).data.cpu().numpy()
            
            action = self._sample_action(p, net, obs_var, self.guides, action_var=action_var, testing=testing)
        else:
            p = None
            action = np.random.rand(self.args.num_total_act) * 2 - 1
        action = np.clip(action, -1, 1)
        guide_act = self.get_guide_action(action)

        if not testing:
            self.prev_act = action
            return action, guide_act
        else:
            self.prev_act = action[0]
            return action, guide_act, p

    def reset(self):
        self.prev_act = np.array([1.0, 0.0])
