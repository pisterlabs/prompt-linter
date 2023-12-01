# -*- coding: utf-8 -*-

# Copyright 2020 Sungho Jeon and Heidelberg Institute for Theoretical Studies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import logging

import w2vEmbReader
from corpus.corpus_base import PAD, BOS, EOS

import models.model_base
import utils
from utils import INT, FLOAT, LONG

#
class Model_EMNLP18_Centt(models.model_base.BaseModel):
    """ class for EMNLP18 implementation
        Title: A Neural Local Coherence Model for Text Quality Assessment
        Ref: https://www.aclweb.org/anthology/D18-1464
    """

    def __init__(self, config, corpus_target, embReader):
        super(Coh_Model_EMNLP18, self).__init__(config)

        ####
        # init parameters
        self.max_num_sents = config.max_num_sents  # document length, in terms of the number of sentences
        self.max_len_sent = config.max_len_sent  # sentence length, in terms of words
        self.max_len_doc = config.max_len_doc  # document length, in terms of words
        self.batch_size =  config.batch_size

        self.vocab = corpus_target.vocab  # word2id
        self.rev_vocab = corpus_target.rev_vocab  # id2word
        self.vocab_size = len(self.vocab)
        # self.go_id = self.rev_vocab[BOS]
        self.pad_id = self.rev_vocab[PAD]
        self.bos_id = self.rev_vocab[BOS]
        self.eos_id = self.rev_vocab[EOS]
        self.num_special_vocab = corpus_target.num_special_vocab

        self.embed_size = config.embed_size
        self.dropout_rate = config.dropout
        self.rnn_cell_size = config.rnn_cell_size
        self.path_pretrained_emb = config.path_pretrained_emb
        self.num_layers = 1
        self.output_size = config.output_size  # the number of final output class

        self.loss_type = config.loss_type
        self.use_gpu = config.use_gpu

        if not hasattr(config, "freeze_step"):
            config.freeze_step = 5000

        ####
        ## model architecture
        self.x_embed = embReader.get_embed_layer()

        # Note that, dropout is disabled for the Readability in the origin impl, although not described in the paper
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        
        self.lstm = nn.LSTM(input_size=self.x_embed.embedding_dim,
                            hidden_size=self.rnn_cell_size,
                            num_layers=self.num_layers,
                            bidirectional=False,
                            dropout=self.dropout_rate,
                            batch_first=True,
                            bias=True)
        self.lstm.apply(self._init_weights)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.leak_relu = nn.LeakyReLU()

        # Convonutional and max-pooling for 5th-stage
        self.conv = nn.Conv1d(in_channels=1,
                  out_channels=1,
                  kernel_size=4,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=True)
        self.max_pooling = nn.MaxPool2d(kernel_size=(1, 3), stride=1)

        # linear layer for final output from coherence vector
        self.linear_1 = nn.Linear(self.max_num_sents - 3, 100, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        self.linear_out_2 = nn.Linear(self.max_num_sents - 3, self.output_size)

        self.linear_out = nn.Linear(100, self.output_size, bias=True)

        if corpus_target.output_bias is not None:  # bias
            init_mean_val = np.expand_dims(corpus_target.output_bias, axis=1)
            bias_val = (np.log(init_mean_val) - np.log(1 - init_mean_val))
            self.linear_out.bias.data = torch.from_numpy(bias_val).type(torch.FloatTensor)
        nn.init.xavier_uniform_(self.linear_out.weight)

        return
    # end __init__
    
    #
    def _init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    #
    def init_hidden_layers(self, batch_size, num_layers, rnn_cell_size):
        hid = torch.autograd.Variable(torch.zeros(num_layers, batch_size, rnn_cell_size))
        hid = utils.cast_type(hid, FLOAT, self.use_gpu)
        nn.init.xavier_uniform_(hid)

        return hid
    # end init_hidden_layers

    # sort multi-dim array by 2d vs 2d dim (e.g., skip batch dim with higher than 2d indices)
    def sort_hid(self, input_unsorted, ind_len_sorted):
        squeezed = input_unsorted.squeeze(0)
        input_sorted = squeezed[ind_len_sorted]
        unsqueezed = input_sorted.unsqueeze(0)

        return unsqueezed
    # end sort_2nd_dim

    #
    def forward(self, text_inputs, mask_input, len_seq, len_sents, tid, len_para=None, mode=""):
        # init hidden layers (teacher forcing)
        last_hid = self.init_hidden_layers(batch_size=self.batch_size, num_layers=self.num_layers,
                                           rnn_cell_size=self.rnn_cell_size)
        last_cell = self.init_hidden_layers(batch_size=self.batch_size, num_layers=self.num_layers,
                                            rnn_cell_size=self.rnn_cell_size)

        ## Stage 1: Embedding representation
        #flat_sent_input = text_inputs.view(-1, self.max_len_sent) #
        #print(text_inputs.shape)
        x_input = self.x_embed(text_inputs)  # 4dim, (batch_size, num_sents, len_sent, embed_size)

        ## Stage 2: Encoding context via LSTM from input sent
        hid_encoded = []  # list of LSTM states from sentences, denoted "H" in the paper
        hid_mask = []  # list for actual length of hidden layer
        mask = mask_input.view(text_inputs.shape)
        for ind_sent in range(self.max_num_sents):
            # embedding representation
            sent_text = text_inputs[:, ind_sent, :].contiguous()
            sent_x_inputs = x_input[:, ind_sent, :].contiguous()  # e.g., 1st sentence of all docs in batch; (batch_size, len_sent, embed_size)
            cur_sents_mask = mask_input[:, ind_sent, :]
            
            len_seq_sorted, ind_len_sorted = torch.sort(len_seq, descending=True)  # ind_len_sorted: (batch_size, num_sents)

            # x_input_sorted = self.sort_by_2d(sent_x_inputs, ind_len_sorted)
            sent_x_input_sorted = sent_x_inputs[ind_len_sorted]

            ## encoding context via LSTM from input sentneces
            # sorting last_hid and last_cell to feed to next training (not needed in nightly version)
            last_hid = self.sort_hid(last_hid, ind_len_sorted) # sort for RNN training
            last_cell = self.sort_hid(last_cell, ind_len_sorted) # sort for RNN training

            # encoding sentence via RNN
            self.lstm.flatten_parameters()
            sent_lstm_out, (last_hid, last_cell) = self.lstm(sent_x_input_sorted, (last_hid, last_cell)) # out: (batch_size, len_sent, cell_size)

            # # Note that, in the nightly version, instead, "enforce_sorted=False" for the automatic sorting ()
            # x_input_packed = pack_padded_sequence(sent_x_input_sorted, lengths=len_seq_sorted, batch_first=True)
            # # x_input_packed = pack_padded_sequence(sent_x_input_sorted, cur_len_sent_sorted, batch_first=True, enforce_sorted=False)
            # out_packed, (last_hid, last_cell) = self.lstm(x_input_packed, (last_hid, last_cell))
            # sent_lstm_out, out_len = pad_packed_sequence(out_packed, batch_first=True)

            # applying mask to last_hid and last_cell by only actual length of sentence
            _, ind_origin = torch.sort(ind_len_sorted)

            sent_lstm_out = sent_lstm_out[ind_origin]
            last_hid = self.sort_hid(last_hid, ind_origin)  # return to origin index for masking
            last_cell = self.sort_hid(last_cell, ind_origin)  # return to origin index for masking

            sent_lstm_out = sent_lstm_out * cur_sents_mask.unsqueeze(2)

            # store encoded sentence
            hid_encoded.append(sent_lstm_out)
        # end for ind_sent

        ## Stage 3: get the most similar hidden states
        vec_close_states = [] # (max_num_sents, batch_size, cell_size)
        for i in range(len(hid_encoded)-1):
            encoded_state_i = hid_encoded[i] # encoded from current sentence
            encoded_state_j = hid_encoded[i+1] # encoded from next sentence

            # get similarity (actually, original paper describes that it is just matrix multiplication)
            sim_states = torch.bmm(encoded_state_i, encoded_state_j.transpose(2,1))  # matmul corresponding to the paper
            sim_states = sim_states.clamp(min=0)  # filter for positive (relu), because it is similarity
            sim_states = self.dropout_layer(sim_states)

            # select two states with maximum similarity
            vec_H = []
            ind_max_sim = torch.argmax(sim_states.view(sim_states.shape[0], -1), dim=1)  # sim index matrix between all states
            for b_id in range(sim_states.shape[0]):
                val_ind = ind_max_sim[b_id]
                max_ind_i = math.floor(val_ind / sim_states.shape[2])
                max_ind_j = val_ind % sim_states.shape[2]

                max_state_i = encoded_state_i[b_id, max_ind_i, :]
                max_state_j = encoded_state_j[b_id, max_ind_j, :]

                vec_ij = (max_state_i + max_state_j) / 2
                vec_H.append(vec_ij)
            # end for batch_id

            vec_H = torch.stack(vec_H) # convert to torch
            vec_close_states.append(vec_H)
        # end for range(len(hid_encoded)), stage3

        ## Stage 4: produce coherence vector
        vec_coh = [] # final output vector represents coherence
        for i in range(len(vec_close_states)-1): # (max_num_sents, batch_size, cell_size)
            vec_u = vec_close_states[i] # (batch_size, cell_size)
            vec_v = vec_close_states[i+1]

            # get similarity (d) between vectors; paper describes that matrix multiplication with division by vector size
            dist_vec_states = torch.mm(vec_u, vec_v.transpose(1, 0))
            dist_states = torch.diag(dist_vec_states)  # they are already the most related states, thus only extract the sim value between them
            dist_states = dist_states / vec_u.shape[1]  # divide by vector size, i.e., cell size
            dist_states = dist_states.clamp(min=0)  # again, only positivity for similarity

            vec_coh.append(dist_states)
        # end for vec_close_states

        vec_coh = torch.stack(vec_coh)  # convert to torch variable from list, (num_sents-2, batch_size)
        vec_coh = vec_coh.permute(1, 0)  # transpose to shape for linear layer (batch * num_sents-2)
        vec_coh = self.dropout_layer(vec_coh)

        # applying convolutional layer and max_pooling
        vec_coh = vec_coh.unsqueeze(1) # nn.conv requires (mini_batch x in_channel x w)
        vec_coh = self.conv(vec_coh)
        vec_coh = self.leak_relu(vec_coh)
        vec_coh = vec_coh.squeeze(1)

        ##
        linear_out_coh = self.linear_1(vec_coh)
        linear_out_coh = self.tanh(linear_out_coh)
        #linear_out_coh = self.dropout_layer(linear_out_coh)
        coh_score = self.linear_out(linear_out_coh)
        
        if self.loss_type.lower() == "mseloss":
            coh_score = self.sigmoid(coh_score)


        outputs = []
        outputs.append(coh_score)

        # return coh_score
        return outputs
    # end forward

# end class
