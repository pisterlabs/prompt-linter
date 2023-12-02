# coding=utf-8
# Copyright (c) 2022, PengCheng Laboratory.  All rights reserved.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# Most of the code here has been copied from:
#   https://github.com/NVIDIA/Megatron-LM/blob/v2.5/megatron/model/bert_model.py
# with some modifications.

"""WaveFormer model."""

import torch

from megatron import get_args
from megatron import mpu
from megatron.model.enums import AttnMaskType
from megatron.model.waveform_model import parallel_gw_logits
from megatron.model.waveform_model import get_waveform_model
from megatron.model import LayerNorm
from megatron.model.utils import openai_gelu, erf_gelu
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from .module import MegatronModule

def gw_extended_attention_mask(attention_mask):
    # We create a 3D attention mask from a 2D tensor mask.
    # [b, 1, s]
    attention_mask_b1s = attention_mask.unsqueeze(1)
    # [b, s, 1]
    attention_mask_bs1 = attention_mask.unsqueeze(2)
    # [b, s, s]
    attention_mask_bss = attention_mask_b1s * attention_mask_bs1
    # [b, 1, s, s]
    extended_attention_mask = attention_mask_bss.unsqueeze(1)

    # Convert attention mask to binary:
    extended_attention_mask = (extended_attention_mask < 0.5)

    return extended_attention_mask

def bert_position_ids(token_ids, dets=1):
    # Create position ids
    token_ids = token_ids[:,:,0]
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids


class GWHead(MegatronModule):
    """Masked GW head for WaveFormer

    Arguments:
        hidden_size: hidden size
        init_method: init method for weight initialization
        layernorm_epsilon: tolerance for layer norm divisions
        parallel_output: whether output logits being distributed or not.
    """

    def __init__(self, hidden_size, init_method,
                 layernorm_epsilon, parallel_output):

        super(GWHead, self).__init__()

        args = get_args()

        self.bias = None
        self.parallel_output = parallel_output

        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)
        self.layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.gelu = torch.nn.functional.gelu
        if args.openai_gelu:
            self.gelu = openai_gelu
        elif args.onnx_safe:
            self.gelu = erf_gelu

    def forward(self, hidden_states, word_embeddings_weight):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        output = parallel_gw_logits(hidden_states,
                                    word_embeddings_weight,
                                    self.parallel_output,
                                    bias=self.bias)
        return output

def post_waveform_model_processing(gw_output, pooled_output,
                                   gw_head, binary_head,
                                   logit_weights,
                                   get_atten_value):
    # Output.
    if not get_atten_value:
        gw_logits = gw_head(
            gw_output, logit_weights) 
    else:
        gw_logits = gw_head(
            gw_output[0], logit_weights) 

    binary_logits = None
    if binary_head is not None:
        binary_logits = binary_head(pooled_output)

    if not get_atten_value:
        return gw_logits, binary_logits
    else:
        return gw_logits, gw_output[-1]


class WaveFormerModel(MegatronModule):
    """Bert Language model."""

    def __init__(self,
                 num_tokentypes=2,
                 add_binary_head=True,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True, 
                 get_atten_value=False):
        super(WaveFormerModel, self).__init__()
        args = get_args()

        self.add_binary_head = add_binary_head
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.dets = len(args.dets.split(','))
        self.get_atten_value = get_atten_value

        init_method = init_method_normal(args.init_method_std)
        scaled_init_method = scaled_init_method_normal(args.init_method_std,
                                                       args.num_layers)

        self.gw_model, self._gw_model_key = get_waveform_model(
            num_tokentypes=num_tokentypes,
            add_pooler=self.add_binary_head,
            encoder_attn_mask_type=AttnMaskType.padding,
            init_method=init_method,
            scaled_init_method=scaled_init_method,
            pre_process=self.pre_process,
            post_process=self.post_process,
            get_atten_value=self.get_atten_value)

        self.initialize_word_embeddings(init_method_normal)

        if self.post_process:
            self.gw_head = GWHead(
                args.hidden_size, init_method, args.layernorm_epsilon, parallel_output)
            self._gw_head_key = 'gw_head'
            self.binary_head = None
            if self.add_binary_head:
                self.binary_head = get_linear_layer(args.hidden_size, 2,
                                                    init_method)
                self._binary_head_key = 'binary_head'

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.gw_model.set_input_tensor(input_tensor)

    def forward(self, bert_model_input, attention_mask,
                tokentype_ids=None, gw_labels=None):

        extended_attention_mask = gw_extended_attention_mask(attention_mask)
        input_ids = bert_model_input
        position_ids = bert_position_ids(input_ids, self.dets)

        gw_output = self.gw_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            tokentype_ids=tokentype_ids
        )

        if self.post_process and self.add_binary_head:
            gw_output, pooled_output = gw_output
        else:
            pooled_output = None

        if self.post_process:
            return post_waveform_model_processing(gw_output, pooled_output,
                                                  self.gw_head, self.binary_head,
                                                  self.word_embeddings_weight(),
                                                  self.get_atten_value)
        else:
            return gw_output


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""

        state_dict_ = {}
        state_dict_[self._gw_model_key] \
            = self.gw_model.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
        if self.post_process:
            state_dict_[self._gw_head_key] \
                = self.gw_head.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        if self.post_process and self.add_binary_head:
            state_dict_[self._binary_head_key] \
                = self.binary_head.state_dict(destination, prefix, keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.gw_model.load_state_dict(
            state_dict[self._gw_model_key], strict=strict)
        if self.post_process:
            self.gw_head.load_state_dict(
                state_dict[self._gw_head_key], strict=strict)
        if self.post_process and self.add_binary_head:
            self.binary_head.load_state_dict(
                state_dict[self._binary_head_key], strict=strict)
        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
