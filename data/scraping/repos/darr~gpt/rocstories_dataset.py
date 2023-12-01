#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : rocstories_dataset.py
# Create date : 2019-03-15 15:16
# Modified date : 2019-03-20 14:43
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import csv
import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data import TensorDataset

from pybase import pylog
from openai_gpt import OpenAIGPTTokenizer
from openai_gpt import get_dataset_path

def _load_rocstories_dataset(dataset_path):
    """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
    with open(dataset_path, encoding='utf_8') as f:
        f = csv.reader(f)
        output = []
        next(f) # skip the first line
        for line in tqdm(f):
            output.append((' '.join(line[1:5]), line[5], line[6], int(line[-1])-1))
    return output

def _tokenize_and_encode(obj, tokenizer):
    """ Tokenize and encode a nested object """
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    elif isinstance(obj, int):
        return obj
    return list(_tokenize_and_encode(o, tokenizer) for o in obj)

def _get_encoded_datasets(tokenizer, config):
    pylog.info("Encoding dataset...")
    train_dataset = _load_rocstories_dataset(config["train_dataset"])
    eval_dataset = _load_rocstories_dataset(config["eval_dataset"])
    datasets = (train_dataset, eval_dataset)
    encoded_datasets = _tokenize_and_encode(datasets, tokenizer)
    return encoded_datasets

def _get_max_and_input_length(model, encoded_datasets):
    max_length = model.config.n_positions // 2 - 2
    pylog.info("max_length:%s" % max_length)

    input_length = max(len(story[:max_length]) + max(len(cont1[:max_length]), len(cont2[:max_length])) + 3  \
                           for dataset in encoded_datasets for story, cont1, cont2, _ in dataset)
    input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model
    pylog.info("input_length:%s" % input_length)
    return max_length, input_length

def _check_or_download_dataset(config):
    _ = get_dataset_path(config)

def _get_tensor_dataset(model, config):
    special_tokens_lt = config["special_tokens_lt"]
    tokenizer = OpenAIGPTTokenizer.from_pretrained(config=config, special_tokens_lt=special_tokens_lt)
    special_tokens_ids = _get_special_token_ids(tokenizer, config)

    _check_or_download_dataset(config)

    # encode the datasets
    encoded_datasets = _get_encoded_datasets(tokenizer, config)
    # Compute the max input length for the Transformer
    max_length, input_length = _get_max_and_input_length(model, encoded_datasets)
    # Prepare inputs tensors and dataloaders
    tensor_datasets = _pre_process_datasets(encoded_datasets, input_length, max_length, *special_tokens_ids)
    return tensor_datasets

def _get_tensor_train_dataset(model, config):
    tensor_datasets = _get_tensor_dataset(model, config)
    train_tensor_dataset = tensor_datasets[0]
    return train_tensor_dataset

def _get_tensor_eval_dataset(model, config):
    tensor_datasets = _get_tensor_dataset(model, config)
    eval_tensor_dataset = tensor_datasets[1]
    return eval_tensor_dataset

def _get_special_token_ids(tokenizer, config):
    special_tokens = config["special_tokens_lt"]
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    return special_tokens_ids

def _pre_process_datasets(encoded_datasets, input_len, cap_length, start_token, delimiter_token, clf_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, 2, input_len), dtype=np.int64)
        mc_token_ids = np.zeros((n_batch, 2), dtype=np.int64)
        lm_labels = np.full((n_batch, 2, input_len), fill_value=-1, dtype=np.int64)
        mc_labels = np.zeros((n_batch,), dtype=np.int64)
        for i, (story, cont1, cont2, mc_label), in enumerate(dataset):
            with_cont1 = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
            with_cont2 = [start_token] + story[:cap_length] + [delimiter_token] + cont2[:cap_length] + [clf_token]
            input_ids[i, 0, :len(with_cont1)] = with_cont1
            input_ids[i, 1, :len(with_cont2)] = with_cont2
            mc_token_ids[i, 0] = len(with_cont1) - 1
            mc_token_ids[i, 1] = len(with_cont2) - 1
            lm_labels[i, 0, :len(with_cont1)-1] = with_cont1[1:]
            lm_labels[i, 1, :len(with_cont2)-1] = with_cont2[1:]
            mc_labels[i] = mc_label
        all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets

def get_train_dataloader(model, config):
    train_tensor_dataset = _get_tensor_train_dataset(model, config)
    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config["train_batch_size"])

    return train_dataloader, train_data

def get_eval_dataloader(model, config):
    eval_tensor_dataset = _get_tensor_eval_dataset(model, config)
    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=config["eval_batch_size"])

    return eval_dataloader
