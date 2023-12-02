#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import RandomSampler
from tqdm import tqdm
from transformers.tokenization_gpt2 import GPT2Tokenizer
from transformers.tokenization_openai import OpenAIGPTTokenizer
from model.seq2seq import TransformerSeq2Seq
from model.seq2seq_vocab import Seq2seqTokenizer

from .loss import LabelSmoothingLoss
from .optim import Adam
from .optim import NoamOpt
from .transformer_model import TransformerModel
from .utils import pad_sequence
from .utils import repeat_along_dim1

SPECIAL_TOKENS = ['<bos>', '<eos>', '<pad>', '<talker1_bos>', '<talker2_bos>', '<talker1_eos>', '<talker2_eos>',
                  '<info_bos>', '<info_eos>', '.', ',', '?', '!', ':']

class Trainer:
    def __init__(self, model, train_dataset, trainer_config, writer, logger=None, test_dataset=None, valid_dataset=None,
                 n_jobs=0, label_smoothing=0, device=torch.device('cuda'), ignore_idxs=[], local_rank=-1,
                 apex_level=None, apex_loss_scale=None, evaluate_full_sequences=False, full_input=False,
                 max_length=511, max_y_length=80, uncertainty_loss=False, new_dataset=False, best_model_path='',
                 extra_module_lr_rate=1, no_persona=False, pointer_gen=False):
        n_gpu = torch.cuda.device_count()
        if logger is None:
            self.logger = logging.getLogger(__file__)
        else:
            self.logger = logger
        self.logger.info("device: {}, distributed training: {}, apex_level: {}, apex_scale_loss: {},  n_gpu: {}".format(
            device, bool(local_rank != -1), apex_level, apex_loss_scale, n_gpu))

        self.train_batch_size = trainer_config.train_batch_size
        self.test_batch_size = trainer_config.test_batch_size
        self.lr = trainer_config.lr
        self.lr_warmup = trainer_config.lr_warmup
        self.weight_decay = trainer_config.weight_decay
        self.batch_split = trainer_config.batch_split
        self.s2s_weight = trainer_config.s2s_weight
        self.lm_weight = trainer_config.lm_weight
        self.risk_weight = trainer_config.risk_weight
        self.hits_weight = trainer_config.hits_weight
        self.single_input = trainer_config.single_input
        self.clip_grad = trainer_config.clip_grad
        self.n_epochs = trainer_config.n_epochs
        self.linear_schedule = trainer_config.linear_schedule
        self.patience = trainer_config.patience
        self.model_saving_interval = trainer_config.model_saving_interval
        self.device = device
        self.ignore_idxs = ignore_idxs
        self.apex_level = apex_level
        self.no_persona = no_persona
        self.evaluate_full_sequences = evaluate_full_sequences
        self.global_step = 0
        self.local_rank = local_rank
        self.full_input = full_input
        self.max_length = max_length
        self.max_y_length = max_y_length
        self.new_dataset = new_dataset
        self.best_ppl = 1e35
        self.best_model_path = best_model_path
        if train_dataset is not None:
            self.negative_samples = train_dataset.negative_samples
        self.model_type = 'pretrain'
        self.patience_cnt = 0
        self.stop_training = False
        self.pointer_gen = pointer_gen

        self.model = model.to(device)
        self.uncertainty_loss = uncertainty_loss

        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=self.model.padding_idx).to(device)
        self.hits_criterion = nn.CrossEntropyLoss().to(device)
        self.criterion = LabelSmoothingLoss(n_labels=self.model.n_embeddings, smoothing=label_smoothing, ignore_index=self.model.padding_idx).to(device)

        param_optimizer = list(self.model.named_parameters())
        # Here we should remove parameters which are not used during to avoid breaking apex with None grads
        self.loss_weight = None
        if self.uncertainty_loss:
            if self.hits_weight > 0:
                loss_weight = torch.zeros(3, device=device)
            else:
                loss_weight = torch.zeros(2, device=device)
            self.loss_weight = ('loss.weight', nn.Parameter(loss_weight))
            param_optimizer.append(self.loss_weight)
        no_decay = ['bias', 'loss']
        if extra_module_lr_rate == 1:
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
        else:
            params, extra_params, no_decay_params, extra_no_decay_params = [], [], [], []
            for n, p in param_optimizer:
                if not any(nd in n for nd in no_decay):
                    if 'attention_module' in n:
                        extra_params.append(p)
                    else:
                        params.append(p)
                else:
                    if 'attention_module' in n:
                        extra_no_decay_params.append(p)
                    else:
                        no_decay_params.append(p)

            optimizer_grouped_parameters = [
                {'params': params, 'weight_decay': self.weight_decay},
                {'params': extra_params, 'weight_decay': self.weight_decay, 'extra': True},
                {'params': no_decay_params, 'weight_decay': 0.0}
            ]
            if len(extra_no_decay_params) != 0:
                optimizer_grouped_parameters.append({'params': extra_no_decay_params, 'weight_decay': 0, 'extra': True})

        base_optimizer = Adam(optimizer_grouped_parameters, lr=self.lr)
        assert local_rank == -1 or apex_level is None, 'Distributed model with apex optimization is not supported right now.'
        # self.model, base_optimizer = apex_model(self.model, optimizer=base_optimizer,
        #                                         apex_level=apex_level, apex_loss_scale=apex_loss_scale)

        if not self.linear_schedule:
            self.optimizer = NoamOpt(self.model.embeddings_size, self.lr_warmup, base_optimizer, lr=self.lr,
                                     linear_schedule=False, apex_level=apex_level, loss_weight=self.loss_weight,
                                     extra_module_lr_rate=extra_module_lr_rate)
        else:
            total_steps = len(train_dataset) * self.n_epochs // self.train_batch_size
            if local_rank != -1:
                total_steps = total_steps // torch.distributed.get_world_size()
            self.optimizer = NoamOpt(self.model.embeddings_size, self.lr_warmup, base_optimizer, linear_schedule=True,
                                     lr=self.lr, total_steps=total_steps, apex_level=apex_level, loss_weight=self.loss_weight,
                                     extra_module_lr_rate=extra_module_lr_rate)

        if local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size // self.batch_split,
                                           sampler=train_sampler,
                                           num_workers=n_jobs, collate_fn=self.collate_func)
        self.train_dataset = train_dataset  # used to sample negative examples
        if test_dataset is not None and local_rank in [-1, 0]:  # only do evaluation on main process
            self.test_dataloader = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False,
                                              num_workers=n_jobs, collate_fn=self.collate_func)
        if valid_dataset is not None and local_rank in [-1, 0]:
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.test_batch_size, shuffle=False,
                                              num_workers=n_jobs, collate_fn=self.collate_func)
        self.vocab = train_dataset.vocab
        self.writer = writer

        if isinstance(self.model, TransformerSeq2Seq):
            self.model_type = 'seq2seq'

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'global_step': self.global_step}

    def load_state_dict(self, state_dict):
        if state_dict.__contains__('model') and state_dict.__contains__('optimizer'):
            self.model.load_state_dict(state_dict['model'], strict=False)
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.global_step = state_dict['global_step']
        else:
            self.model.load_state_dict(state_dict, strict=False)

    def collate_func(self, data):
        persona_info, h, y, distractors_batch = zip(*data)

        contexts = []

        if max(map(len, persona_info)) > 0:
            persona_info = [torch.tensor(d, dtype=torch.long) for d in persona_info]
            contexts.append(persona_info)

        if max(map(len, h)) > 0:
            h = [torch.tensor(d, dtype=torch.long) for d in h]
            contexts.append(h)

        y_out = [torch.tensor(d, dtype=torch.long) for d in y]

        distractors = [torch.tensor(d, dtype=torch.long) for distractors in distractors_batch for d in distractors]

        if self.single_input:
            # we concatenate all the contexts in y (idem for distractors)
            if self.no_persona:
                for c in contexts[1]:
                    c[0][0] = self.vocab.bos_id
                y_out = [torch.cat(pieces, dim=0) for pieces in zip(*([contexts[1]] + [y_out]))]
                distractors_contexts = []
                for c in contexts[1]:
                    distractors_contexts.extend([c] * self.negative_samples)
                distractors = [torch.cat(pieces, dim=0) for pieces in zip(*([distractors_contexts] + [distractors]))]
                lengths = [(contexts[1][i].size(0), y_out[i].size(0)) for i in range(len(y_out))]
                lengths.extend([(distractors_contexts[i].size(0), distractors[i].size(0)) for i in range(len(distractors))])
                contexts = lengths
            else:
                distractors_contexts = [[], []]
                for i in range(len(contexts[0])):
                    distractors_contexts[0].extend([contexts[0][i]] * self.negative_samples)
                    distractors_contexts[1].extend([contexts[1][i]] * self.negative_samples)
                if self.model_type == 'seq2seq':
                    y_out1 = [torch.cat(pieces, dim=0) for pieces in zip(*(contexts))]
                    distractors1 = [torch.cat(pieces, dim=0) for pieces in zip(*(distractors_contexts))]
                    lengths = [(contexts[0][i].size(0) + contexts[1][i].size(0), y_out[i].size(0)) for i in range(len(y_out))]
                    if len(distractors1) > 0:
                        lengths.extend(
                            [(distractors_contexts[0][i].size(0) + distractors_contexts[1][i].size(0),
                              distractors[i].size(0))
                             for i in range(len(distractors1))])
                    y_out = (y_out1, y_out)
                    distractors = (distractors1, distractors)
                else:
                    y_out = [torch.cat(pieces, dim=0) for pieces in zip(*(contexts + [y_out]))]
                    distractors = [torch.cat(pieces, dim=0) for pieces in zip(*(distractors_contexts + [distractors]))]
                    lengths = [(contexts[0][i].size(0) + contexts[1][i].size(0), y_out[i].size(0)) for i in range(len(y_out))]
                    if len(distractors) > 0:
                        lengths.extend(
                            [(distractors_contexts[0][i].size(0) + distractors_contexts[1][i].size(0),
                              distractors[i].size(0))
                                    for i in range(len(distractors))])
                contexts = lengths
            # extended_contexts = [[t for t in c for _ in range(len(distractors)//len(y))] for c in contexts]
            # distractors = [torch.cat(pieces, dim=0) for pieces in zip(*(extended_contexts + [distractors]))]
            # if self.no_persona:
            #     contexts = contexts[1]
            # else:
            #     contexts = [torch.cat(pieces, dim=0) for pieces in zip(*contexts)]
        else:
            if self.full_input:
                y_out = [torch.cat(pieces, dim=0) for pieces in zip(*(contexts + [y_out]))]
                extended_contexts = [[t for t in c for _ in range(len(distractors) // len(y))] for c in contexts]
                distractors = [torch.cat(pieces, dim=0) for pieces in zip(*(extended_contexts + [distractors]))]
                for i, seq in enumerate(y_out):
                    if seq.shape[0] > self.max_length:
                        history_start, history_end = -1, -1
                        for j in range(seq.shape[0]):
                            if history_start == -1 and \
                                    (seq[j][1] == self.vocab.talker1_dialog_id or seq[j][1] == self.vocab.talker2_dialog_id):
                                history_start = j
                            if history_end == -1 and seq[j][1] == self.vocab.sent_dialog_id:
                                history_end = j
                                break
                        history_length = self.max_length - history_start - (seq.shape[0] - history_end)
                        y_out[i] = torch.cat([y_out[i][:history_start], y_out[i][history_end - history_length:]], dim=0)

            # Pad now so we pad correctly when we have only a single input (context concatenated with y)
        if self.single_input:
            if isinstance(y_out, tuple):
                y_out = ([y[-(self.max_length - 1):] for y in y_out[0]], [y[:(self.max_y_length - 1)] for y in y_out[1]])
                distractors = ([d[-(self.max_length - 1):] for d in distractors[0]],
                               [d[:(self.max_length - 1)] for d in distractors[1]])
            else:
                y_out = [y[-(self.max_length - 1):] for y in y_out]
                distractors = [d[-(self.max_length - 1):] for d in distractors]
            contexts = [c if c[1] <= self.max_length - 1 else (c[0] - (c[1] - self.max_length + 1), self.max_length - 1) for c in contexts]
        else:
            y_out = [y[: self.max_length] for y in y_out]
            distractors = [d[: self.max_length] for d in distractors]
            contexts = [c[:self.max_length] for c in contexts]
        # with open('error1.pickle', 'wb') as f:
        #     pickle.dump({'y_out': y_out, 'distractors': distractors}, f)
        if isinstance(y_out, tuple):
            y_out = (pad_sequence(y_out[0], batch_first=True, padding_value=self.model.padding_idx),
                     pad_sequence(y_out[1], batch_first=True, padding_value=self.model.padding_idx))
            distractors = (pad_sequence(distractors[0], batch_first=True, padding_value=self.model.padding_idx),
                           pad_sequence(distractors[1], batch_first=True, padding_value=self.model.padding_idx))
        else:
            y_out = pad_sequence(y_out, batch_first=True, padding_value=self.model.padding_idx)
            distractors = pad_sequence(distractors, batch_first=True, padding_value=self.model.padding_idx)
        if not self.single_input:
            contexts = [pad_sequence(c, batch_first=True, padding_value=self.model.padding_idx) for c in contexts]

        return contexts, y_out, distractors

    # def _lm_loss(self, enc_generated, context):
    #
    #     #ignore_mask = torch.stack([context == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
    #     #context.masked_fill_(ignore_mask, self.model.padding_idx)
    #     #prevs = enc_generated[:, :-1, :].contiguous()
    #     #nexts = context[:, 1:].contiguous() if context.dim() == 2 else context[:, 1:, 0].contiguous()
    #     #return self.lm_criterion(prevs.view(-1, prevs.shape[-1]).float(), nexts.view(-1))
    #     return self.lm_criterion(enc_generated.view(-1, enc_generated.shape[-1]), context[:,:,0].view(-1))

    def _lm_loss(self, contexts, enc_contexts):
        batch_lm_loss = torch.tensor(0, dtype=torch.float, device=self.device)

        if self.single_input:
            return batch_lm_loss

        for context in contexts:
            enc_context = self.model.encode(context.clone())
            enc_contexts.append(enc_context)

            if self.lm_weight > 0:
                context_outputs = self.model.generate(enc_context[0])
                ignore_mask = torch.stack([context == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                context.masked_fill_(ignore_mask, self.model.padding_idx)
                prevs = context_outputs[:, :-1, :].contiguous()
                nexts = context[:, 1:].contiguous() if context.dim() == 2 else context[:, 1:, 0].contiguous()
                batch_lm_loss += self.lm_criterion(prevs.view(-1, prevs.shape[-1]).float(), nexts.view(-1)) / len(contexts)
        return batch_lm_loss

    def _loss_single(self, targets, distractors, lengths):
        input_ids = targets[:, :, 0].contiguous()
        token_type_ids = targets[:, :, 1].contiguous()
        lm_labels = -100 * torch.ones_like(input_ids)
        mc_token_ids = torch.tensor([l[1] - 1 for l in lengths], device=self.device)
        cur_batch_size = input_ids.size(0)
        for i in range(cur_batch_size):
            lm_labels[i, lengths[i][0] + 1: lengths[i][1]] = targets[i, lengths[i][0] + 1: lengths[i][1], 0].contiguous()
        lm_loss, lm_logits, mc_logits, _ = self.model(input_ids, token_type_ids=token_type_ids, lm_labels=lm_labels,
                mc_token_ids=mc_token_ids[: cur_batch_size])
        all_mc_logits = [mc_logits.unsqueeze(-1)]
        if distractors.size()[0] > 0:
            for i in range(self.negative_samples):
                distractor_ids = distractors[cur_batch_size * i: cur_batch_size * (i + 1), :, 0]. contiguous()
                distractor_type_ids = distractors[cur_batch_size * i: cur_batch_size * (i + 1), :, 1]. contiguous()
                distractor_mix_replace_batch = None
                _, mc_logits, _ = self.model(
                    distractor_ids,
                    token_type_ids=distractor_type_ids,
                    mc_token_ids=mc_token_ids[cur_batch_size * (i + 1): cur_batch_size * (i + 2)])
                all_mc_logits.append(mc_logits.unsqueeze(-1))
        mc_labels = torch.zeros_like(mc_logits, dtype=torch.long)
        mc_logits = torch.cat(all_mc_logits, dim=-1)
        if self.model.training:
            loss_fct = CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits, mc_labels)
        else:
            mc_loss = torch.sum(torch.max(mc_logits, dim=1)[1] == mc_labels).float() / mc_labels.shape[0]
        return lm_loss, mc_loss

    def _s2s_loss(self, targets, enc_contexts, negative_samples):
        hidden_state, padding_mask = None, None

        nexts = targets[:, 1:].contiguous() if targets.dim() == 2 else targets[:, 1:, 0].contiguous()
        if self.hits_weight > 0 and negative_samples > 0:
            # Keep the hidden states for hits@1 loss
            outputs, hidden_state, padding_mask = self.model.decode(targets[:, :-1].contiguous(), enc_contexts)
        else:
            if isinstance(self.model, TransformerModel):
                outputs = self.model.decode(targets[:, :-1].contiguous(), enc_contexts)
            else:
                outputs, _, _ = self.model.decode(targets[:, :-1].contiguous(), enc_contexts)
        if self.full_input:
            for i in range(targets.shape[0]):
                for j in range(targets.shape[1]):
                    if targets[i][j][1] == self.vocab.sent_dialog_id:
                        nexts[i][: j] = self.model.padding_idx
                        break

        outputs = outputs.view(-1, outputs.shape[-1]).float()
        nexts = nexts.view(-1)

        loss = self.criterion(F.log_softmax(outputs, dim=-1), nexts) if self.model.training \
               else self.lm_criterion(outputs, nexts)
        return loss, hidden_state, padding_mask

    def _hist(self, distractors, hidden_state, padding_mask, enc_contexts, negative_samples):
        batch_hits_loss = torch.tensor(0, dtype=torch.float, device=self.device)

        if self.hits_weight == 0 or negative_samples == 0:
            return batch_hits_loss

        extended_contexts = repeat_along_dim1(enc_contexts, negative_samples)
        neg_logits = self.model.decode_classify(distractors, extended_contexts)
        true_logits = self.model.classify(hidden_state, padding_mask)
        clf_logits = torch.cat((true_logits.view(-1, 1), neg_logits.view(-1, negative_samples)), dim=1)
        clf_labels = torch.tensor([0] * len(true_logits), dtype=torch.long, device=self.device)

        batch_hits_loss = self.hits_criterion(clf_logits.float(), clf_labels) if self.model.training else \
                          torch.sum(torch.max(clf_logits, dim=1)[1] == clf_labels).float() / clf_labels.shape[0]

        return batch_hits_loss

    def _risk_loss(self, contexts, targets, enc_contexts, risk_func):

        if risk_func is None or self.risk_weight == 0:
            return torch.tensor(0, dtype=torch.float, device=self.device)

        self.model.eval()  # desactivate dropout

        if self.single_input:
            beam_starts = pad_sequence(contexts, batch_first=True, padding_value=self.model.padding_idx, left=True)
            beams, beam_lens = self.model.beam_search(beam_starts=beam_starts, return_beams=True)
        else:
            beams, beam_lens = self.model.beam_search(enc_contexts=enc_contexts, return_beams=True)

        self.model.train()  # re-activate dropout

        labels = targets if targets.dim() == 2 else targets[:, :, 0]
        labels_lens = labels.ne(self.model.padding_idx).sum(dim=-1)
        labels_start = [context.shape[0] + 1 for context in contexts] if self.single_input else [1] * len(labels)
        labels = [t[s:l - 1].tolist() for t, s, l in zip(labels, labels_start, labels_lens)]

        batch_risks = []
        for b in range(self.model.beam_size):
            predictions = [t[b][1:l[b] - 1].tolist() for t, l in zip(beams, beam_lens)]
            risks = torch.tensor(risk_func(predictions, labels), dtype=torch.float, device=self.device)
            batch_risks.append(risks)
        batch_risks = torch.stack(batch_risks, dim=-1)

        if self.model.dialog_embeddings:
            beams = torch.stack((beams, torch.full_like(beams, self.model.sent_dialog_id)), dim=beams.dim())

        if self.single_input:
            start = beam_starts.size(1)
            beam_starts.unsqueeze_(1)
            beam_starts = beam_starts.repeat([1, self.model.beam_size] + [1] * len(beam_starts.size()[2:])) # tail_dims for dialog_embeddings
            beams = torch.cat((beam_starts, beams), dim=2)

        batch_probas = []
        for b in range(self.model.beam_size):
            inputs = beams[:, b, :-1]
            outputs = beams[:, b, 1:]

            outputs = outputs[:, :, 0] if outputs.dim() == 3 else outputs
            logits = self.model.decode(inputs, enc_contexts)

            probas = F.log_softmax(logits.float(), dim=-1)
            probas = torch.gather(probas, -1, outputs.unsqueeze(-1)).squeeze(-1)
            probas.masked_fill_(outputs.eq(self.model.padding_idx), 0)
            probas = probas[:, start:] if self.single_input else probas

            probas = probas.sum(dim=-1) / beam_lens[:, b].float()

            batch_probas.append(probas)
        batch_probas = torch.stack(batch_probas, dim=-1)
        batch_probas = F.softmax(batch_probas, dim=-1)

        batch_risk_loss = torch.mean((batch_risks * batch_probas).sum(dim=-1))

        return batch_risk_loss

    def optimizer_step(self, lm_loss, risk_loss, hits_loss, s2s_loss, full_loss):
        if self.clip_grad is not None:
            for group in self.optimizer.param_groups:
                nn.utils.clip_grad_norm_(group['params'], self.clip_grad)

        self.optimizer.step()
        self.optimizer.zero_grad()

        global_step = max(self.global_step, 0)
        self.writer.add_scalar("training/lm_loss", lm_loss, global_step=global_step)
        self.writer.add_scalar("training/risk_loss", risk_loss, global_step=global_step)
        self.writer.add_scalar("training/hits_loss", hits_loss, global_step=global_step)
        self.writer.add_scalar("training/s2s_loss", s2s_loss, global_step=global_step)
        self.writer.add_scalar("training/full_loss", full_loss, global_step=global_step)
        self.writer.add_scalar("training/lr", self.optimizer.get_lr(), global_step=global_step)

        self.global_step += 1

    def _eval_train(self, epoch, risk_func=None): # add ppl and hits@1 evaluations
        self.model.train()

        tqdm_data = tqdm(self.train_dataloader, desc='Train (epoch #{})'.format(epoch))
        s2s_loss = 0
        lm_loss = 0
        risk_loss = 0
        hits_loss = 0
        for i, (contexts, targets, distractors) in enumerate(tqdm_data):
            negative_samples = self.negative_samples
            if not self.single_input:
                contexts, targets, distractors = [c.to(self.device) for c in contexts], targets.to(self.device), \
                                                 distractors.to(self.device)
                enc_contexts = []

                if isinstance(self.model, TransformerSeq2Seq):
                    loss = self.model(contexts, targets)
                    full_loss = (loss / self.batch_split,)
                    s2s_loss = (i * s2s_loss + loss.item()) / (i + 1)
                    tqdm_data.set_postfix({'s2s_loss': s2s_loss})
                else:
                    # lm loss on contexts
                    batch_lm_loss = self._lm_loss(contexts, enc_contexts)
                    # batch_lm_loss = (self._lm_loss(enc_persona_generated, persona.clone()) + self._lm_loss(enc_dialog_generated, dialog.clone())) / 2

                    # s2s loss on targets
                    batch_s2s_loss, hidden_state, padding_mask = self._s2s_loss(targets, enc_contexts, negative_samples)

                    # hits@1 loss on distractors and targets
                    batch_hits_loss = self._hist(distractors, hidden_state, padding_mask, enc_contexts, negative_samples)

                    # risk loss
                    batch_risk_loss = self._risk_loss(contexts, targets, enc_contexts, risk_func)
                    full_loss = (self.lm_weight * batch_lm_loss / self.batch_split,
                                 self.risk_weight * batch_risk_loss / self.batch_split,
                                 self.hits_weight * batch_hits_loss / self.batch_split,
                                 self.s2s_weight * batch_s2s_loss / self.batch_split)
                    lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1)
                    s2s_loss = (i * s2s_loss + batch_s2s_loss.item()) / (i + 1)
                    risk_loss = (i * risk_loss + batch_risk_loss.item()) / (i + 1)
                    hits_loss = (i * hits_loss + batch_hits_loss.item()) / (i + 1)
                    tqdm_data.set_postfix({'lm_loss': lm_loss, 's2s_loss': s2s_loss,
                                           'risk_loss': risk_loss, 'hits_loss': hits_loss})
            else:
                if isinstance(self.model, TransformerSeq2Seq):
                    input_ids, labels, lengths = targets[0].to(self.device), targets[1].to(self.device), contexts
                    input_ids_replace, labels_replace = None, None,
                    loss = self.model(input_ids, labels, input_ids_replace, labels_replace)
                    if isinstance(loss, tuple):
                        full_loss = (loss[0] / self.batch_split, )
                        s2s_loss = (i * s2s_loss + loss[1].item()) / (i + 1)
                    else:
                        full_loss = (loss / self.batch_split, )
                        s2s_loss = (i * s2s_loss + loss.item()) / (i + 1)
                    tqdm_data.set_postfix({'s2s_loss': s2s_loss})
                else:
                    targets, distractors, lengths = targets.to(self.device), distractors.to(self.device), contexts

                    batch_s2s_loss, batch_hits_loss = self._loss_single(targets, distractors, lengths)
                    full_loss = (self.s2s_weight * batch_s2s_loss / self.batch_split,
                                 self.hits_weight * batch_hits_loss / self.batch_split)
                    s2s_loss = (i * s2s_loss + batch_s2s_loss.item()) / (i + 1)
                    hits_loss = (i * hits_loss + batch_hits_loss.item()) / (i + 1)
                    tqdm_data.set_postfix({'s2s_loss': s2s_loss, 'hits_loss': hits_loss})

            # optimization
            full_loss = tuple(filter(lambda x: x.requires_grad, full_loss))
            full_loss = self.optimizer.backward(full_loss)
            if self.pointer_gen and (torch.isnan(self.model.generator.p_gen_linear._parameters['weight']._grad[0][0]) or \
                torch.isinf(self.model.generator.p_gen_linear._parameters['weight']._grad[0][0])):
                self.optimizer.zero_grad()
                self.logger.info('Abnormal gradient')
            if (i + 1) % self.batch_split == 0:
                self.optimizer_step(lm_loss, risk_loss, hits_loss, s2s_loss, full_loss)
        if (i + 1) % self.batch_split != 0:
            self.optimizer_step(lm_loss, risk_loss, hits_loss, s2s_loss, full_loss)

    def _get_eval_loss(self, contexts, targets, distractors, metrics, index):
        lengths, enc_contexts = None, []
        if self.single_input:
            if isinstance(self.model, TransformerSeq2Seq):
                input_ids, labels, lengths = targets[0].to(self.device), targets[1].to(self.device), contexts
                batch_s2s_loss = self.model(input_ids, labels)
                if isinstance(batch_s2s_loss, tuple):
                    batch_s2s_loss = batch_s2s_loss[1]
                batch_hits_acc = torch.tensor(0, dtype=torch.float)
            else:
                targets, distractors, lengths = targets.to(self.device), distractors.to(self.device), contexts
                batch_s2s_loss, batch_hits_acc = self._loss_single(targets, distractors, lengths, None, None)
        else:
            contexts, targets, distractors = [c.to(self.device) for c in contexts], targets.to(self.device), \
                                             distractors.to(self.device)
            if isinstance(self.model, TransformerSeq2Seq):
                batch_s2s_loss, enc_contexts = self.model(contexts, targets, return_encoded=True)
                batch_hits_acc = torch.tensor(0, dtype=torch.float)
            else:

                # lm loss
                batch_lm_loss = self._lm_loss(contexts, enc_contexts)
                metrics['lm_loss'] = (metrics['lm_loss'] * index + batch_lm_loss.item()) / (index + 1)

                # s2s loss on targets
                batch_s2s_loss, hidden_state, padding_mask = self._s2s_loss(targets, enc_contexts,
                                                                            self.negative_samples)
                # hits@1 loss on distractors and targets
                batch_hits_acc = self._hist(distractors, hidden_state, padding_mask,
                                            enc_contexts, self.negative_samples)
                metrics['lm_ppl'] = (metrics['lm_ppl'] * index + math.exp(batch_lm_loss)) / (index + 1)

        metrics['s2s_loss'] = (metrics['s2s_loss'] * index + batch_s2s_loss.item()) / (index + 1)
        metrics['hits_acc'] = (metrics['hits_acc'] * index + batch_hits_acc.item()) / (index + 1)
        metrics['s2s_ppl'] = (metrics['s2s_ppl'] * index + math.exp(batch_s2s_loss)) / (index + 1)
        return metrics, lengths, enc_contexts

    def _get_eval_predictions(self, contexts, targets, lengths, enc_contexts, metrics, metric_funcs,
                             external_metrics_func, index):
        string_references, string_predictions = [], []
        if self.evaluate_full_sequences:
            if self.single_input:
                if isinstance(self.model, TransformerSeq2Seq):
                    labels = targets[1]
                else:
                    labels = []
                    for i in range(targets.shape[0]):
                        labels.append(targets[i, lengths[i][0] + 1: lengths[i][1], 0])
                    labels = pad_sequence(labels, batch_first=True, padding_value=self.model.padding_idx)
            elif not self.full_input:
                labels = targets if targets.dim() == 2 else targets[:, :, 0]
            else:
                labels = []
                new_targets = []
                for i in range(targets.shape[0]):
                    label_start, label_end = -1, targets.shape[1]
                    for j in range(targets.shape[1]):
                        if targets[i][j][1] == self.model.sent_dialog_id and label_start == -1:
                            label_start = j
                        if targets[i][j][1] == self.model.padding_idx:
                            label_end = j
                            break
                    labels.append(targets[i, label_start: label_end, 0])
                    new_targets.append(targets[i, : label_start])
                labels = pad_sequence(labels, batch_first=True, padding_value=self.model.padding_idx)
                targets = pad_sequence(new_targets, batch_first=True, padding_value=self.model.padding_idx,
                                       left=True)
            if self.single_input:
                predictions = []
                if isinstance(self.model, TransformerSeq2Seq):
                    input_ids = targets[0].to(self.device)
                    predictions = self.model.inference(input_ids)
                else:
                    for i in range(targets.size(0)):
                        input_ids = targets[i, :lengths[i][0] + 1, 0].to(self.device)
                        token_type_ids = targets[i, :lengths[i][0] + 1, 1].to(self.device)
                        prediction = self.model.inference(input_ids, token_type_ids)
                        predictions.append(prediction)
            else:
                if isinstance(self.model, TransformerSeq2Seq):
                    predictions = self.model.inference(contexts, encoder_outputs=enc_contexts)
                else:
                    if self.full_input:
                        predictions = self.model.inference(beam_starts=targets, enc_contexts=enc_contexts)
                    else:
                        predictions = self.model.inference(enc_contexts=enc_contexts)

            labels_lens = labels.ne(self.model.padding_idx).sum(dim=-1)
            if not self.single_input:
                labels_start = [1] * len(targets)
                labels = [t[s:l - 1].tolist() for t, s, l in zip(labels, labels_start, labels_lens)]
            else:
                labels = [t[: l - 1].tolist() for t, l in zip(labels, labels_lens)]

            for name, func in metric_funcs.items():
                score = func(predictions, labels)
                metrics[name] = (metrics[name] * index + score) / (index + 1)

            if external_metrics_func:
                # Store text strings for external metrics
                if isinstance(self.vocab, OpenAIGPTTokenizer) or isinstance(self.vocab, GPT2Tokenizer) or \
                        isinstance(self.vocab, Seq2seqTokenizer):
                    string_references = list(
                        self.vocab.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in
                        labels)
                    string_predictions = list(
                        self.vocab.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in
                        predictions)
                else:
                    string_references = list(self.vocab.ids2string(t) for t in labels)
                    string_predictions = list(self.vocab.ids2string(t) for t in predictions)
                string_predictions = [x.replace('\n', ' ') for x in string_predictions]
        return string_predictions, string_references

    def _eval_test(self, metric_funcs={}, external_metrics_func=None, epoch=-1, inference=False, is_best=False,
                   raw_entail_data=None):
        with torch.no_grad():
            self.model.eval()
            if epoch == -1:
                tqdm_data = tqdm(self.test_dataloader, desc='Test')
                self.logger.info('Starting testing on Test dataset')
            else:
                tqdm_data = tqdm(self.valid_dataloader, desc='Test')
                self.logger.info('Starting testing on Valid dataset')
            metrics = {name: 0 for name in ('s2s_loss', 'lm_loss', 'hits_acc', 'lm_ppl', 's2s_ppl') + tuple(metric_funcs.keys())}
            full_predictions, full_references = [], []
            for i, (contexts, targets, distractors) in enumerate(tqdm_data):
                '''Get the loss, ppl for each batch'''
                metrics, lengths, enc_contexts = self._get_eval_loss(contexts, targets, distractors, metrics, i)
                # full sequence loss
                cur_predictions, cur_references = self._get_eval_predictions(contexts, targets, lengths, enc_contexts,
                             metrics, metric_funcs, external_metrics_func, i)
                full_predictions.extend(cur_predictions)
                full_references.extend(cur_references)
                tqdm_data.set_postfix(dict(**metrics))

            if external_metrics_func and self.evaluate_full_sequences:
                external_metrics = external_metrics_func(full_references, full_predictions, epoch, is_best)
                metrics.update(external_metrics)

            # logging
            global_step = max(self.global_step, 0)
            if self.writer is not None:
                for key, value in metrics.items():
                    self.writer.add_scalar("eval/{}".format(key), value, global_step=global_step)
            self.logger.info(metrics)

            if epoch != -1:
                if metrics['s2s_ppl'] < self.best_ppl:
                    self.logger.info('Current ppl BEATS the previous best one, previous best is %.5f', self.best_ppl)
                    self.best_ppl = metrics['s2s_ppl']
                    torch.save(self.model.state_dict(), self.best_model_path)
                else:
                    self.patience_cnt += 1
                    self.logger.info('Current ppl CANNOT BEATS the previous best one, previous best is %.5f', self.best_ppl)
                    if self.patience > 0 and self.patience_cnt > self.patience:
                        self.stop_training = True
            if epoch % self.model_saving_interval == 0 and epoch >= self.model_saving_interval and \
                    self.model_type in ['seq2seq']:
                torch.save(self.model.state_dict(), self.best_model_path + '_' + str(epoch))

    def _build_split_data_list(self, targets, distractors, lengths, distractor_lengths, split_batch_size):
        split_targets, split_distractors, split_lengths = [], [], []
        batch_size = targets.size(0)
        i = 0
        while i * split_batch_size < batch_size:
            split_targets.append(targets[i * split_batch_size: (i + 1) * split_batch_size])
            split_distractors.append(distractors[split_batch_size * i * self.negative_samples: split_batch_size * (
                        i + 1) * self.negative_samples])
            split_lengths.append(lengths[split_batch_size * i: split_batch_size * (i + 1)] +
                                 distractor_lengths[split_batch_size * i * self.negative_samples: split_batch_size * (
                                             i + 1) * self.negative_samples])
            i += 1
        return split_targets, split_distractors, split_lengths

    def _concat(self, xs):
        return torch.cat([x.view(-1) for x in xs])

    def _clip_grad_norm(self, grads, max_norm, norm_type=2):
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        if norm_type == float('inf'):
            total_norm = max(grad.data.abs().max() for grad in grads)
        else:
            total_norm = 0
            for grad in grads:
                grad_norm = grad.data.norm(norm_type)
                total_norm += grad_norm ** norm_type
            total_norm = total_norm ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for grad in grads:
                grad.data.mul_(clip_coef)
        return total_norm

    def test(self, metric_funcs={}, external_metrics_func=None, epoch=-1, inference=False):
        if hasattr(self, 'valid_dataloader') or hasattr(self, 'test_dataloader'):
            self._eval_test(metric_funcs, external_metrics_func, epoch, inference)
            if epoch == -1 and not inference:
                self.logger.info('Loading the best model...')
                if os.path.exists(self.best_model_path):
                    state_dict = torch.load(self.best_model_path, map_location=self.device)
                    if state_dict.__contains__('model'):
                        self.model.load_state_dict(state_dict['model'], strict=False)
                    else:
                        self.model.load_state_dict(state_dict)
                    self._eval_test(metric_funcs, external_metrics_func, epoch, inference, is_best=True)
                else:
                    self.logger.warning('The PPL of current model is higher than the initial threshold, '
                                        'no test stage will be executed.')

    def train(self, after_epoch_funcs=[], risk_func=None):
        for epoch in range(1, self.n_epochs + 1):
            self.logger.info('===============================')
            self.logger.info('Start training on Epoch %d', epoch)
            self._eval_train(epoch, risk_func)
            # self._eval_test()

            for func in after_epoch_funcs:
                func(epoch)
            self.logger.info('End training on Epoch %d', epoch)
            self.logger.info('===============================')
            if self.stop_training:
                self.logger.info('Training will be STOPPED in advance due to exceeding patience number')
                break

        for func in after_epoch_funcs:
            func(-1)
