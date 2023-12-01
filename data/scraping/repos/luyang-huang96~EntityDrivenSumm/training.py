""" module providing basic training utilities"""
from coherence_interface.coherence_inference import coherence_infer, batch_global_infer
import os
from os.path import join
from time import time
from datetime import timedelta
from itertools import starmap

from cytoolz import curry, reduce, concat

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
import tensorboardX
from utils import PAD, UNK, START, END
from metric import compute_rouge_l, compute_rouge_n, compute_rouge_l_summ
from nltk import sent_tokenize



def get_basic_grad_fn(net, clip_grad, max_grad=1e2):
    def f():
        grad_norm = clip_grad_norm_(
            [p for p in net.parameters() if p.requires_grad], clip_grad)
        try:
            grad_norm = grad_norm.item()
        except AttributeError:
            grad_norm = grad_norm
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log = {}
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f

def get_loss_args(net_out, bw_args):
    if isinstance(net_out, tuple):
        loss_args = net_out + bw_args
    else:
        loss_args = (net_out, ) + bw_args
    return loss_args

@curry
def compute_loss(net, criterion, fw_args, loss_args):
    net_out = net(*fw_args)
    all_loss_args = get_loss_args(net_out, loss_args)
    loss = criterion(*all_loss_args)
    return loss

@curry
def val_step(loss_step, fw_args, loss_args):
    loss = loss_step(fw_args, loss_args)
    try:
        n_data = loss.size(0)
        return n_data, loss.sum().item()
    except RuntimeError:
        n_data = 1
        return n_data, loss.item()

@curry
def basic_validate(net, criterion, val_batches):
    print('running validation ... ', end='')
    net.eval()
    start = time()
    with torch.no_grad():
        validate_fn = val_step(compute_loss(net, criterion))
        n_data, tot_loss = reduce(
            lambda a, b: (a[0]+b[0], a[1]+b[1]),
            starmap(validate_fn, val_batches),
            (0, 0)
        )
    val_loss = tot_loss / n_data
    print(
        'validation finished in {}                                    '.format(
            timedelta(seconds=int(time()-start)))
    )
    print('validation loss: {:.4f} ... '.format(val_loss))
    return {'loss': val_loss}

@curry
def rl_validate(net, val_batches, coherence_func=None, coh_coef = 0.01, local_coh_func=None, local_coh_coef=0.005):
    print('running validation ... ', end='')
    def argmax(arr, keys):
        return arr[max(range(len(arr)), key=lambda i: keys[i].item())]
    def sum_id2word(raw_article_sents, decs, attns):
        dec_sents = []
        for i, raw_words in enumerate(raw_article_sents):
            dec = []
            for id_, attn in zip(decs, attns):
                if id_[i] == END:
                    break
                elif id_[i] == UNK:
                    dec.append(argmax(raw_words, attn[i]))
                else:
                    dec.append(id2word[id_[i].item()])
            dec_sents.append(dec)
        return dec_sents
    net.eval()
    start = time()
    i = 0
    score = 0
    score_coh = 0
    score_local_coh = 0
    with torch.no_grad():
        for fw_args, bw_args in val_batches:
            raw_articles = bw_args[0]
            id2word = bw_args[1]
            raw_targets = bw_args[2]
            greedies, greedy_attns = net.greedy(*fw_args)
            greedy_sents = sum_id2word(raw_articles, greedies, greedy_attns)
            bl_scores = []
            if coherence_func is not None:
                bl_coh_scores = []
                bl_coh_inputs = []
            if local_coh_func is not None:
                bl_local_coh_scores = []
            for baseline, target in zip(greedy_sents, raw_targets):
                bss = sent_tokenize(' '.join(baseline))
                if coherence_func is not None:
                    bl_coh_inputs.append(bss)
                    # if len(bss) > 1:
                    #     input_args = (bss,) + coherence_func
                    #     coh_score = coherence_infer(*input_args) / 2
                    # else:
                    #     coh_score = 0
                    # bl_coh_scores.append(coh_score)
                if local_coh_func is not None:
                    local_coh_score = local_coh_func(bss)
                    bl_local_coh_scores.append(local_coh_score)
                bss = [bs.split(' ') for bs in bss]
                tgs = sent_tokenize(' '.join(target))
                tgs = [tg.split(' ') for tg in tgs]
                bl_score = compute_rouge_l_summ(bss, tgs)
                bl_scores.append(bl_score)
            # print('blscore:', bl_score)
            # print('baseline:', bss)
            # print('target:', tgs)

            bl_scores = torch.tensor(bl_scores, dtype=torch.float32, device=greedy_attns[0].device)
            if coherence_func is not None:
                input_args = (bl_coh_inputs,) + coherence_func
                bl_coh_scores = batch_global_infer(*input_args)
                bl_coh_scores = torch.tensor(bl_coh_scores, dtype=torch.float32, device=greedy_attns[0].device)
                score_coh += bl_coh_scores.mean().item() * 100 * coh_coef
            if local_coh_func is not None:
                bl_local_coh_scores = torch.tensor(bl_local_coh_scores, dtype=torch.float32, device=greedy_attns[0].device)
                score_local_coh += bl_local_coh_scores.mean().item() * local_coh_coef * 100
            reward = bl_scores.mean().item()
            i += 1
            score += reward * 100

    val_score = score / i
    if coherence_func is not None:
        val_coh_score = score_coh / i
    else:
        val_coh_score = 0
    val_local_coh_score = 0
    print(
        'validation finished in {}                                    '.format(
            timedelta(seconds=int(time()-start)))
    )
    print('validation reward: {:.4f} ... '.format(val_score))
    if coherence_func is not None:
        print('validation reward: {:.4f} ... '.format(val_coh_score))
    if local_coh_func is not None:
        val_local_coh_score = score_local_coh / i
        print('validation {} reward: {:.4f} ... '.format(local_coh_func.__name__, val_local_coh_score))
    print('n_data:', i)
    return {'score': val_score + val_coh_score + val_local_coh_score}

class BasicPipeline(object):
    def __init__(self, name, net,
                 train_batcher, val_batcher, batch_size,
                 val_fn, criterion, optim, grad_fn=None):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._criterion = criterion
        self._opt = optim
        # grad_fn is calleble without input args that modifyies gradient
        # it should return a dictionary of logging values
        self._grad_fn = grad_fn
        self._val_fn = val_fn

        self._n_epoch = 0  # epoch not very useful?
        self._batch_size = batch_size
        self._batches = self.batches()

    def batches(self):
        while True:
            for fw_args, bw_args in self._train_batcher(self._batch_size):
                yield fw_args, bw_args
            self._n_epoch += 1

    def get_loss_args(self, net_out, bw_args):
        if isinstance(net_out, tuple):
            loss_args = net_out + bw_args
        else:
            loss_args = (net_out, ) + bw_args
        return loss_args

    def train_step(self):
        # forward pass of model
        self._net.train()
        #self._net.zero_grad()
        fw_args, bw_args = next(self._batches)
        net_out = self._net(*fw_args)

        # get logs and output for logging, backward
        log_dict = {}
        loss_args = self.get_loss_args(net_out, bw_args)

        # backward and update ( and optional gradient monitoring )
        loss = self._criterion(*loss_args).mean()
        loss.backward()
        log_dict['loss'] = loss.item()
        if self._grad_fn is not None:
            log_dict.update(self._grad_fn())
        self._opt.step()
        self._net.zero_grad()
        torch.cuda.empty_cache()

        return log_dict

    def validate(self):
        return self._val_fn(self._val_batcher(self._batch_size))

    def checkpoint(self, save_path, step, val_metric=None):
        save_dict = {}
        if val_metric is not None:
            name = 'ckpt-{:6f}-{}'.format(val_metric, step)
            save_dict['val_metric'] = val_metric
        else:
            name = 'ckpt-{}'.format(step)

        save_dict['state_dict'] = self._net.state_dict()
        save_dict['optimizer'] = self._opt.state_dict()
        torch.save(save_dict, join(save_path, name))

    def terminate(self):
        self._train_batcher.terminate()
        self._val_batcher.terminate()

class AbsSelfCriticalPipeline(object):
    def __init__(self, name, net,
                 train_batcher, val_batcher, batch_size,
                 val_fn, optim, grad_fn=None, coh_fn=None, coh_coef=0.01, local_coh_func=None, local_coh_coef=0.005):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._opt = optim
        # grad_fn is calleble without input args that modifyies gradient
        # it should return a dictionary of logging values
        self._grad_fn = grad_fn
        self._val_fn = val_fn
        self._coh_fn = coh_fn

        self._n_epoch = 0  # epoch not very useful?
        self._batch_size = batch_size
        self._batches = self.batches()
        self._coh_cof = coh_coef
        self._local_coh_fun = local_coh_func
        self._local_co_coef = local_coh_coef
        self._weights = [0., 0.5, 0.5]

    def batches(self):
        while True:
            for fw_args, bw_args in self._train_batcher(self._batch_size):
                yield fw_args, bw_args
            self._n_epoch += 1

    def get_loss_args(self, net_out, bw_args):
        if isinstance(net_out, tuple):
            loss_args = net_out + bw_args
        else:
            loss_args = (net_out, ) + bw_args
        return loss_args


    def train_step(self, sample_time=1):
        def argmax(arr, keys):
            return arr[max(range(len(arr)), key=lambda i: keys[i].item())]
        def sum_id2word(raw_article_sents, decs, attns, id2word):
            dec_sents = []
            for i, raw_words in enumerate(raw_article_sents):
                dec = []
                for id_, attn in zip(decs, attns):
                    if id_[i] == END:
                        break
                    elif id_[i] == UNK:
                        dec.append(argmax(raw_words, attn[i]))
                    else:
                        dec.append(id2word[id_[i].item()])
                dec_sents.append(dec)
            return dec_sents
        def pack_seq(seq_list):
            return torch.cat([_.unsqueeze(1) for _ in seq_list], 1)
        # forward pass of model
        self._net.train()
        #self._net.zero_grad()
        fw_args, bw_args = next(self._batches)
        raw_articles = bw_args[0]
        id2word = bw_args[1]
        raw_targets = bw_args[2]
        with torch.no_grad():
            greedies, greedy_attns = self._net.greedy(*fw_args)
        greedy_sents = sum_id2word(raw_articles, greedies, greedy_attns, id2word)
        bl_scores = []
        if self._coh_fn is not None:
            bl_coh_scores = []
            bl_coh_inputs = []
        if self._local_coh_fun is not None:
            bl_local_coh_scores = []
        for baseline, target in zip(greedy_sents, raw_targets):
            bss = sent_tokenize(' '.join(baseline))
            tgs = sent_tokenize(' '.join(target))
            if self._coh_fn is not None:
                bl_coh_inputs.append(bss)
                # if len(bss) > 1:
                #     input_args = (bss, ) + self._coh_fn
                #     coh_score = coherence_infer(*input_args) / 2
                # else:
                #     coh_score = 0
                # bl_coh_scores.append(coh_score)
            if self._local_coh_fun is not None:
                local_coh_score = self._local_coh_fun(bss)
                bl_local_coh_scores.append(local_coh_score)
            bss = [bs.split(' ') for bs in bss]
            tgs = [tg.split(' ') for tg in tgs]
            #bl_score = compute_rouge_l_summ(bss, tgs)
            bl_score = (self._weights[2] * compute_rouge_l_summ(bss, tgs) + \
                        self._weights[0] * compute_rouge_n(list(concat(bss)), list(concat(tgs)), n=1) + \
                        self._weights[1] * compute_rouge_n(list(concat(bss)), list(concat(tgs)), n=2))
            bl_scores.append(bl_score)
        bl_scores = torch.tensor(bl_scores, dtype=torch.float32, device=greedy_attns[0].device)
        if self._coh_fn is not None:
            input_args = (bl_coh_inputs,) + self._coh_fn
            bl_coh_scores = batch_global_infer(*input_args)
            bl_coh_scores = torch.tensor(bl_coh_scores, dtype=torch.float32, device=greedy_attns[0].device)
        if self._local_coh_fun is not None:
            bl_local_coh_scores = torch.tensor(bl_local_coh_scores, dtype=torch.float32, device=greedy_attns[0].device)

        # print('bl:', bl_scores)
        # print('bl_coh:', bl_coh_scores)

        for _ in range(sample_time):
            samples, sample_attns, seqLogProbs = self._net.sample(*fw_args)
            sample_sents = sum_id2word(raw_articles, samples, sample_attns, id2word)
            sp_seqs = pack_seq(samples)
            _masks = (sp_seqs > PAD).float()
            sp_seqLogProb = pack_seq(seqLogProbs)
            #loss_nll = - sp_seqLogProb.squeeze(2)
            loss_nll = - sp_seqLogProb.squeeze(2) * _masks.detach().type_as(sp_seqLogProb)
            sp_scores = []
            if self._coh_fn is not None:
                sp_coh_scores = []
                sp_coh_inputs = []
            if self._local_coh_fun is not None:
                sp_local_coh_scores = []
            for sample, target in zip(sample_sents, raw_targets):
                sps = sent_tokenize(' '.join(sample))
                tgs = sent_tokenize(' '.join(target))
                if self._coh_fn is not None:
                    sp_coh_inputs.append(sps)
                    # if len(sps) > 1:
                    #     input_args = (sps,) + self._coh_fn
                    #     coh_score = coherence_infer(*input_args) / 2
                    # else:
                    #     coh_score = 0
                    # sp_coh_scores.append(coh_score)
                if self._local_coh_fun is not None:
                    local_coh_score = self._local_coh_fun(sps)
                    sp_local_coh_scores.append(local_coh_score)
                sps = [sp.split(' ') for sp in sps]
                tgs = [tg.split(' ') for tg in tgs]
                #sp_score = compute_rouge_l_summ(sps, tgs)
                sp_score = (self._weights[2] * compute_rouge_l_summ(sps, tgs) + \
                            self._weights[0] * compute_rouge_n(list(concat(sps)), list(concat(tgs)), n=1) + \
                            self._weights[1] * compute_rouge_n(list(concat(sps)), list(concat(tgs)), n=2))
                sp_scores.append(sp_score)
            sp_scores = torch.tensor(sp_scores, dtype=torch.float32, device=greedy_attns[0].device)
            reward = sp_scores.view(-1, 1) - bl_scores.view(-1, 1)
            reward.requires_grad_(False)
            if self._coh_fn is not None:
                input_args = (sp_coh_inputs,) + self._coh_fn
                sp_coh_scores = batch_global_infer(*input_args)
                sp_coh_scores = torch.tensor(sp_coh_scores, dtype=torch.float32, device=greedy_attns[0].device)
                reward_coh = sp_coh_scores.view(-1, 1) - bl_coh_scores.view(-1, 1)
                reward_coh.requires_grad_(False)
                reward = reward + self._coh_cof * reward_coh
            if self._local_coh_fun is not None:
                sp_local_coh_scores = torch.tensor(sp_local_coh_scores, dtype=torch.float32, device=greedy_attns[0].device)
                reward_local = sp_local_coh_scores.view(-1, 1) - bl_local_coh_scores.view(-1, 1)
                reward_local.requires_grad_(False)
                reward = reward + self._local_co_coef * reward_local

            if _ == 0:
                loss = reward.contiguous().detach() * loss_nll
                loss = loss.sum()
                full_length = _masks.data.float().sum()
            else:
                loss += (reward.contiguous().detach() * loss_nll).sum()
                full_length += _masks.data.float().sum()

        # print('sp:', sp_scores)
        # print('sp_coh:', sp_coh_scores)
        loss = loss / full_length
        # backward and update ( and optional gradient monitoring )
        loss.backward()
        log_dict = {}
        if self._coh_fn is not None:
            log_dict['reward'] = bl_scores.mean().item() + self._coh_cof * bl_coh_scores.mean().item()
        else:
            log_dict['reward'] = bl_scores.mean().item()
        if self._grad_fn is not None:
            log_dict.update(self._grad_fn())
        self._opt.step()
        self._net.zero_grad()
        torch.cuda.empty_cache()

        return log_dict

    def validate(self):
        return self._val_fn(self._val_batcher(self._batch_size))

    def checkpoint(self, save_path, step, val_metric=None):
        save_dict = {}
        if val_metric is not None:
            name = 'ckpt-{:6f}-{}'.format(val_metric, step)
            save_dict['val_metric'] = val_metric
        else:
            name = 'ckpt-{}'.format(step)

        save_dict['state_dict'] = self._net.state_dict()
        save_dict['optimizer'] = self._opt.state_dict()
        torch.save(save_dict, join(save_path, name))

    def terminate(self):
        self._train_batcher.terminate()
        self._val_batcher.terminate()


class BasicTrainer(object):
    """ Basic trainer with minimal function and early stopping"""
    def __init__(self, pipeline, save_dir, ckpt_freq, patience,
                 scheduler=None, val_mode='loss'):
        assert isinstance(pipeline, BasicPipeline) or isinstance(pipeline, AbsSelfCriticalPipeline)
        assert val_mode in ['loss', 'score']
        self._pipeline = pipeline
        self._save_dir = save_dir
        self._logger = tensorboardX.SummaryWriter(join(save_dir, 'log'))
        os.makedirs(join(save_dir, 'ckpt'))

        self._ckpt_freq = ckpt_freq
        self._patience = patience
        self._sched = scheduler
        self._val_mode = val_mode

        self._step = 0
        self._running_loss = None
        # state vars for early stopping
        self._current_p = 0
        self._best_val = None

    def log(self, log_dict):
        loss = log_dict['loss'] if 'loss' in log_dict else log_dict['reward']
        if self._running_loss is not None:
            self._running_loss = 0.99*self._running_loss + 0.01*loss
        else:
            self._running_loss = loss
        print('train step: {}, {}: {:.4f}\r'.format(
            self._step,
            'loss' if 'loss' in log_dict else 'reward',
            self._running_loss), end='')
        for key, value in log_dict.items():
            self._logger.add_scalar(
                '{}_{}'.format(key, self._pipeline.name), value, self._step)

    def validate(self):
        print()
        val_log = self._pipeline.validate()
        for key, value in val_log.items():
            self._logger.add_scalar(
                'val_{}_{}'.format(key, self._pipeline.name),
                value, self._step
            )
        if 'reward' in val_log:
            val_metric = val_log['reward']
        else:
            val_metric = (val_log['loss'] if self._val_mode == 'loss'
                          else val_log['score'])
        return val_metric

    def checkpoint(self):
        val_metric = self.validate()
        self._pipeline.checkpoint(
            join(self._save_dir, 'ckpt'), self._step, val_metric)
        if isinstance(self._sched, ReduceLROnPlateau):
            self._sched.step(val_metric)
        else:
            self._sched.step()
        stop = self.check_stop(val_metric)
        return stop

    def check_stop(self, val_metric):
        if self._best_val is None:
            self._best_val = val_metric
        elif ((val_metric < self._best_val and self._val_mode == 'loss')
              or (val_metric > self._best_val and self._val_mode == 'score')):
            self._current_p = 0
            self._best_val = val_metric
        else:
            self._current_p += 1
        return self._current_p >= self._patience

    def train(self):
        try:
            start = time()
            print('Start training')
            while True:
                log_dict = self._pipeline.train_step()
                self._step += 1
                self.log(log_dict)

                if self._step % self._ckpt_freq == 0:
                    stop = self.checkpoint()
                    if stop:
                        break
            print('Training finised in ', timedelta(seconds=time()-start))
        finally:
            self._pipeline.terminate()
