#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : eval_graph.py
# Create date : 2019-03-15 21:32
# Modified date : 2019-03-22 11:44
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from tqdm import tqdm, trange
import torch

from openai_gpt import OpenAIGPTDoubleHeadsModel

from . import rocstories_dataset
from . import func

def _eval_the_model(model, eval_dataloader, config):
    device = config["device"]
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels = batch
        with torch.no_grad():
            _, mc_loss = model(input_ids, mc_token_ids, lm_labels, mc_labels)
            _, mc_logits = model(input_ids, mc_token_ids)

        mc_logits = mc_logits.detach().cpu().numpy()
        mc_labels = mc_labels.to('cpu').numpy()
        tmp_eval_accuracy = func.accuracy(mc_logits, mc_labels)

        eval_loss += mc_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    return eval_loss, eval_accuracy

def _load_finetuned_model(model_config, config):
    device = config["device"]
    output_model_file = func.get_output_model_file_full_path(config)
    model_state_dict = torch.load(output_model_file)
    model = OpenAIGPTDoubleHeadsModel(model_config)
    model.load_state_dict(model_state_dict)
    model.to(device)
    return model

def do_eval(model_config, config):
    model = _load_finetuned_model(model_config, config)
    eval_dataloader = rocstories_dataset.get_eval_dataloader(model, config)
    eval_loss, eval_accuracy = _eval_the_model(model, eval_dataloader, config)
    return eval_loss, eval_accuracy
