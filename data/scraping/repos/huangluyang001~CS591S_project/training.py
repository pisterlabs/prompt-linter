""" module providing basic training utilities"""
#from coherence_interface.coherence_inference import coherence_infer, batch_global_infer
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
import logging
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.tokenization_utils').disabled = True
logging.basicConfig(level=logging.ERROR)
try:
    from apex import amp
except:
    logging.warning('apex not found, fp16 disabled')

from eval_dialog import bleu, compute_f1



def get_basic_grad_fn(net, clip_grad, max_grad=1e2, fp16=False):
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
    except:
        n_data = 1
        return n_data, loss.item()

@curry
def multitask_val_step(loss_step, fw_args, loss_args):
    losses = loss_step(fw_args, loss_args)
    try:
        n_data = losses[0].size(0)
        return n_data, losses[0].sum().item(), losses[1].sum().item()
    except IndexError:
        n_data = 1
        return n_data, losses[0].item(), losses[1].item()

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
def multitask_validate(net, criterion, val_batches):
    print('running validation ... ', end='')
    net.eval()
    start = time()
    with torch.no_grad():
        validate_fn = multitask_val_step(compute_loss(net, criterion))
        n_data, tot_loss, tot_aux_loss = reduce(
            lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2]),
            starmap(validate_fn, val_batches),
            (0, 0, 0)
        )
    val_loss = tot_loss / n_data
    val_aux_loss = tot_aux_loss / n_data
    print(
        'validation finished in {}                                    '.format(
            timedelta(seconds=int(time()-start)))
    )
    print('validation loss: {:.4f} ... '.format(val_loss))
    print('validation auxilary loss: {:.4f} ... '.format(val_aux_loss))
    return {'loss': val_loss, 'aux_loss': val_aux_loss}


@curry
def rl_validate(net, val_batches, reward_func=None, reward_coef = 0.01, local_coh_func=None, local_coh_coef=0.005, bert=False, _bleu=False, f1=False):
    print('running validation ... ', end='')
    if bert:
        tokenizer = net._bert_model._tokenizer
        end = tokenizer.encoder[tokenizer._eos_token]
        unk = tokenizer.encoder[tokenizer._unk_token]
    def argmax(arr, keys):
        return arr[max(range(len(arr)), key=lambda i: keys[i].item())]
    def sum_id2word(raw_article_sents, decs, attns):
        if bert:
            dec_sents = []
            for i, raw_words in enumerate(raw_article_sents):
                dec = []
                for id_, attn in zip(decs, attns):
                    if id_[i] == end:
                        break
                    elif id_[i] == unk:
                        dec.append(argmax(raw_words, attn[i]))
                    else:
                        dec.append(id2word[id_[i].item()])
                dec_sents.append(dec)
        else:
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
    score_reward = 0
    score_local_coh = 0
    score_r2 = 0
    score_r1 = 0
    score_bleu = 0
    score_f1 = 0
    bl_r2 = []
    bl_r1 = []
    bl_bleu = []
    bl_f1 = []
    with torch.no_grad():
        for fw_args, bw_args in val_batches:
            raw_articles = bw_args[0]
            id2word = bw_args[1]
            raw_targets = bw_args[2]
            if reward_func is not None:
                questions = bw_args[3]
            greedies, greedy_attns = net.greedy(*fw_args)
            greedy_sents = sum_id2word(raw_articles, greedies, greedy_attns)
            bl_scores = []
            if reward_func is not None:
                bl_coh_inputs = []
            if local_coh_func is not None:
                bl_local_coh_scores = []
            for baseline, target in zip(greedy_sents, raw_targets):
                if bert:
                    text = ''.join(baseline)
                    baseline = bytearray([tokenizer.byte_decoder[c] for c in text]).decode('utf-8',
                                                                                                 errors=tokenizer.errors)
                    baseline = baseline.split(' ')
                    text = ''.join(target)
                    target = bytearray([tokenizer.byte_decoder[c] for c in text]).decode('utf-8',
                                                                                               errors=tokenizer.errors)
                    target = target.split(' ')


                bss = sent_tokenize(' '.join(baseline))
                if reward_func is not None:
                    bl_coh_inputs.append(bss)

                bss = [bs.split(' ') for bs in bss]
                tgs = sent_tokenize(' '.join(target))
                tgs = [tg.split(' ') for tg in tgs]

                bss_bleu = list(concat(bss))
                tgs_bleu = list(concat(tgs))
                bss_bleu = ' '.join(bss_bleu)
                tgs_bleu = ' '.join(tgs_bleu)

                bleu_scores = bleu(bss_bleu, tgs_bleu)
                bleu_score = (bleu_scores[0] + bleu_scores[1] + bleu_scores[2] + bleu_scores[3])
                f1_score = compute_f1(bss_bleu, tgs_bleu)
                bl_bleu.append(bleu_score)
                bl_f1.append(f1_score)
                bl_score = compute_rouge_l_summ(bss, tgs)
                bl_r1.append(compute_rouge_n(list(concat(bss)), list(concat(tgs)), n=1))
                bl_r2.append(compute_rouge_n(list(concat(bss)), list(concat(tgs)), n=2))
                bl_scores.append(bl_score)
            bl_scores = torch.tensor(bl_scores, dtype=torch.float32, device=greedy_attns[0].device)

            if reward_func is not None:
                bl_reward_scores = reward_func.score(questions, bl_coh_inputs)
                bl_reward_scores = torch.tensor(bl_reward_scores, dtype=torch.float32, device=greedy_attns[0].device)
                score_reward += bl_reward_scores.mean().item() * 100

            reward = bl_scores.mean().item()
            i += 1
            score += reward * 100
            score_r2 += torch.tensor(bl_r2, dtype=torch.float32, device=greedy_attns[0].device).mean().item() * 100
            score_r1 += torch.tensor(bl_r1, dtype=torch.float32, device=greedy_attns[0].device).mean().item() * 100
            score_bleu += torch.tensor(bl_bleu, dtype=torch.float32, device=greedy_attns[0].device).mean().item() * 100
            score_f1 += torch.tensor(bl_f1, dtype=torch.float32, device=greedy_attns[0].device).mean().item() * 100

    if _bleu:
        val_score = score_bleu / i
    elif f1:
        val_score = score_f1 / i
    else:
        val_score = score / i
    score_r2 = score_r2 / i
    score_r1 = score_r1 / i
    score_bleu = score_bleu / i
    score_f1 = score_f1 / i
    if reward_func is not None:
        val_reward_score = score_reward / i
    else:
        val_reward_score = 0
    val_local_coh_score = 0
    print(
        'validation finished in {}                                    '.format(
            timedelta(seconds=int(time()-start)))
    )
    print('validation reward: {:.4f} ... '.format(val_score))
    print('validation r2: {:.4f} ... '.format(score_r2))
    print('validation r1: {:.4f} ... '.format(score_r1))
    print('validation bleu: {:4f} ...'.format(score_bleu))
    print('validation f1: {:4f} ...'.format(score_f1))


    if reward_func is not None:
        print('validation reward: {:.4f} ... '.format(val_reward_score))
    if local_coh_func is not None:
        val_local_coh_score = score_local_coh / i
        print('validation {} reward: {:.4f} ... '.format(local_coh_func.__name__, val_local_coh_score))
    print('n_data:', i)
    return {'score': val_score,
            'score_reward:': val_reward_score}

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
        #torch.cuda.empty_cache()

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
                 val_fn, optim, grad_fn=None, weights=[0., 0.5, 0.5], _bleu=False, f1=False):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._opt = optim
        # grad_fn is calleble without input args that modifyies gradient
        # it should return a dictionary of logging values
        self._grad_fn = grad_fn
        self._val_fn = val_fn
        self._w8 = weights

        self._n_epoch = 0  # epoch not very useful?
        self._batch_size = batch_size
        self._batches = self.batches()

        self._bleu = _bleu
        self.f1 = f1

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

        for baseline, target in zip(greedy_sents, raw_targets):
            bss = sent_tokenize(' '.join(baseline))
            tgs = sent_tokenize(' '.join(target))
            bss = [bs.split(' ') for bs in bss]
            tgs = [tg.split(' ') for tg in tgs]
            bss_bleu = list(concat(bss))
            tgs_bleu = list(concat(tgs))
            bss_bleu = ' '.join(bss_bleu)
            tgs_bleu = ' '.join(tgs_bleu)
            #bl_score = compute_rouge_l_summ(bss, tgs)
            if self._bleu:
                bleu_scores = bleu(bss_bleu, tgs_bleu)
                bleu_score = (bleu_scores[0] + bleu_scores[1] + bleu_scores[2] + bleu_scores[3])
                bl_score = bleu_score
            elif self.f1:
                bl_score = compute_f1(bss_bleu, tgs_bleu)
            else:
                bl_score = (self._w8[2] * compute_rouge_l_summ(bss, tgs) + \
                       self._w8[0] * compute_rouge_n(list(concat(bss)), list(concat(tgs)), n=1) + \
                        self._w8[1] * compute_rouge_n(list(concat(bss)), list(concat(tgs)), n=2))
            bl_scores.append(bl_score)
        bl_scores = torch.tensor(bl_scores, dtype=torch.float32, device=greedy_attns[0].device)


        samples, sample_attns, seqLogProbs = self._net.sample(*fw_args)
        sample_sents = sum_id2word(raw_articles, samples, sample_attns, id2word)
        sp_seqs = pack_seq(samples)
        _masks = (sp_seqs > PAD).float()
        sp_seqLogProb = pack_seq(seqLogProbs)
        #loss_nll = - sp_seqLogProb.squeeze(2)
        loss_nll = - sp_seqLogProb.squeeze(2) * _masks.detach().type_as(sp_seqLogProb)
        sp_scores = []

        for sample, target in zip(sample_sents, raw_targets):
            sps = sent_tokenize(' '.join(sample))
            tgs = sent_tokenize(' '.join(target))
            sps = [sp.split(' ') for sp in sps]
            tgs = [tg.split(' ') for tg in tgs]
            #sp_score = compute_rouge_l_summ(sps, tgs)
            sps_bleu = list(concat(sps))
            tgs_bleu = list(concat(tgs))
            sps_bleu = ' '.join(sps_bleu)
            tgs_bleu = ' '.join(tgs_bleu)
            # bl_score = compute_rouge_l_summ(bss, tgs)
            if self._bleu:
                bleu_scores = bleu(sps_bleu, tgs_bleu)
                bleu_score = (bleu_scores[0] + bleu_scores[1] + bleu_scores[2] + bleu_scores[3])
                sp_score = bleu_score
            elif self.f1:
                sp_score = compute_f1(sps_bleu, tgs_bleu)
            else:
                sp_score = (self._w8[2] * compute_rouge_l_summ(sps, tgs) + \
                        self._w8[0] * compute_rouge_n(list(concat(sps)), list(concat(tgs)), n=1) + \
                        self._w8[1]* compute_rouge_n(list(concat(sps)), list(concat(tgs)), n=2))
            sp_scores.append(sp_score)
        sp_scores = torch.tensor(sp_scores, dtype=torch.float32, device=greedy_attns[0].device)

        reward = sp_scores.view(-1, 1) - bl_scores.view(-1, 1)

        reward.requires_grad_(False)
        loss = reward.contiguous().detach() * loss_nll
        loss = loss.sum()
        full_length = _masks.data.float().sum()
        loss = loss / full_length


        loss.backward()

        log_dict = {}

        log_dict['reward'] = bl_scores.mean().item()

        if self._grad_fn is not None:
            log_dict.update(self._grad_fn())
        self._opt.step()
        self._net.zero_grad()
        #torch.cuda.empty_cache()

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
        if 'question_reward' in log_dict:
            print('train step: {}, {}: {:.4f}, {}: {:.4f}, {}: {:.4f}, {}: {:.4f}\r'.format(
                self._step,
                'reward',
                log_dict['reward'],
                'question_reward',
                log_dict['question_reward'],
                'sample_question_reward',
                log_dict['sample_question_reward'],
                'sample_reward',
                log_dict['sample_reward']
                ), end='')
        else:
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

