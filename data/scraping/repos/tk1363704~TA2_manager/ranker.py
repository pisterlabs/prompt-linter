import json
import os
import random
import re
import time
from typing import List, Dict, Union

import numpy as np
import openai
import torch
from retry.api import retry_call

from sentence_transformers import SentenceTransformer, util
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, HoulsbyConfig

import prompt_chatgpt

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print("device is {}".format(device))

class Ranker(nn.Module):

    def __init__(self, model, tokenizer, effective_adapter_name, informative_adapter_name):
        super(Ranker, self).__init__()

        self.model = model
        self.tokenizer = tokenizer
        adapter_config = HoulsbyConfig() #AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2)
        self.model.add_adapter(effective_adapter_name, config=adapter_config)
        self.model.add_adapter(informative_adapter_name, config=adapter_config)
        self.cos = nn.CosineSimilarity(dim=1)

    def load_adapter(self, adapter_path):
        self.model.load_adapter(adapter_path)

    def forward(self, context, violation, rank_list, context_attn, violation_attn, rank_list_attn, adapter_name):
        self.model.set_active_adapters(adapter_name)

        context_states = self.model(input_ids=context, attention_mask = context_attn).last_hidden_state.mean(1)
        violation_states = self.model(input_ids=violation, attention_mask = violation_attn).last_hidden_state.mean(1)
        rank_list_states = self.model(input_ids=rank_list, attention_mask = rank_list_attn).last_hidden_state.mean(1)

        rank_list_scores = self.cos(context_states+violation_states, rank_list_states)

        return rank_list_scores

PROJECT_ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
effective_adapter_name = 'effective_ranker'
informative_adapter_name = 'informative_ranker'

effective_adapter_path = PROJECT_ABSOLUTE_PATH + '/effective_ranker'
informative_adapter_path = PROJECT_ABSOLUTE_PATH + '/informative_ranker'

embedder = prompt_chatgpt.embedder

ranker_backbone_model = embedder._first_module().auto_model

ranker_tokenizer = embedder._first_module().tokenizer

best_model = Ranker(ranker_backbone_model, ranker_tokenizer, effective_adapter_name, informative_adapter_name).to(device)
best_model.load_adapter(effective_adapter_path)
best_model.load_adapter(informative_adapter_path)


def _tensorize_batch(examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]], tokenizer
                     ) -> torch.Tensor:
    def _flatten(l):
        return [item for sublist in l for item in sublist]

    # In order to accept both lists of lists and lists of Tensors
    if isinstance(examples[0], (list, tuple)):
        flatten_list = _flatten(examples)
        # print(flatten_list)
        examples = [torch.tensor(e, dtype=torch.long) for e in flatten_list]
    # print(examples[0].size())
    length_of_first = examples[0].size(0)
    # print(length_of_first)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)
    else:
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have one."
            )
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)


def rank_remediations(dialogue, remediation_list, justification_list):
    context_str = dialogue[-3:-1]
    if len(context_str) == 0:
        context_str = "[PAD]"
    else:
        context_str = " ".join(context_str)

    violation_str = dialogue[-1]
    ranker_list_str = remediation_list


    context = best_model.tokenizer.convert_tokens_to_ids(best_model.tokenizer.tokenize(context_str))
    violation = best_model.tokenizer.convert_tokens_to_ids(best_model.tokenizer.tokenize(violation_str))
    ranker_list = [best_model.tokenizer.convert_tokens_to_ids(best_model.tokenizer.tokenize(candidate)) for candidate in
                   ranker_list_str]

    context = [context] * len(ranker_list)
    violation = [violation] * len(ranker_list)

    context = _tensorize_batch(examples=[context], tokenizer=best_model.tokenizer)
    violation = _tensorize_batch(examples=[violation], tokenizer=best_model.tokenizer)

    rank_list = _tensorize_batch(examples=[ranker_list], tokenizer=best_model.tokenizer)

    context_attn = (context != best_model.tokenizer.pad_token_id)  # src
    violation_attn = (violation != best_model.tokenizer.pad_token_id)  # tgt
    rank_list_attn = (rank_list != best_model.tokenizer.pad_token_id)  # tgt
    effective_scores = best_model(context.to(device), violation.to(device), rank_list.to(device),
                            context_attn.to(device), violation_attn.to(device),
                            rank_list_attn.to(device), effective_adapter_name)
    informative_scores = best_model(context.to(device), violation.to(device), rank_list.to(device),
                            context_attn.to(device), violation_attn.to(device),
                            rank_list_attn.to(device), informative_adapter_name)
    #print(effective_scores)
    #print(informative_scores)
    all_scores = (effective_scores + informative_scores)/2
    # get the index of the max log-probability
    pred = all_scores.argmax().item()
    # get the remediation from the ranker_list_str given the index
    pred_remediation = ranker_list_str[pred]
    pred_justification = justification_list[pred]

    return pred_remediation, pred_justification




