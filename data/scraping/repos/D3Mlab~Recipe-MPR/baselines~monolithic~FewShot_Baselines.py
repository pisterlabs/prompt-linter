from helper import *
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OPTForCausalLM, GPT2Tokenizer
from sklearn.utils import shuffle
from torch import optim
from tqdm import tqdm
import time
import json
import csv
import random
import numpy as np
import pandas as pd
from helper import get_model_and_tokenizer

def few_shot_prompt(q, add_answer=False):
    text = "input: " + q["query"] + " \n"
    text += "output: "
    if add_answer:
        text += q['options'][q['answer']] + " \n"

    return text


def get_fewshot_NLL_score(model, tokenizer, condition, text, filler=' so I recommend ', normalize=False):
    text = condition + filler + text 
    encodings = tokenizer(text, return_tensors="pt") # input to model
    condition = tokenizer(condition, return_tensors="pt")
    stride = condition.input_ids.size(1)

    nlls = []

    begin_loc = 0
    end_loc = stride
    trg_len = encodings.input_ids.size(1) - stride
    input_ids = encodings.input_ids.to('cuda')
    target_ids = input_ids.clone()

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs[0]
        logits = outputs[1].cpu()
        pred = np.argmax(logits, axis=-1)[0]

    if normalize:
        with torch.no_grad():
            c_input_ids = condition.input_ids.to('cuda')
            outputs = model(c_input_ids, labels=c_input_ids)
            c_neg_log_likelihood = outputs[0]
        return (-1 * neg_log_likelihood) - (-1 * c_neg_log_likelihood)
    else:
        return -1 * neg_log_likelihood


def fewshot(dataset, model, tokenizer, normalize, prompt):


    predictions = []

    output_message = ""

    # loop through each query
    for sample in dataset:
        # count += 1
        # if count % 50 == 0:
        #     print('--> ', count)

        # for key in sample['query_type']:
        #     if sample['query_type'][key] == 1:
        #         type_count[key] += 1

        # output_message += str(count) + ' Query: ' + sample["query"] + ' \n'

        scores = []
        q_text = sample["query"]
        p = few_shot_prompt(sample,False)
        q_text = prompt + p 
        for key in sample["options"]:

            score = get_fewshot_NLL_score(model, tokenizer, q_text, sample["options"][key], normalize=normalize, filler='')
            assert not torch.isnan(score), 'score is nan'

            scores.append([key,score])
            # if key == sample["answer"]:
            #     output_message += 'Answer: ' + str(score) + ' ' + sample["options"][key] + ' \n'
            # else:
            #     output_message += str(score) + ' ' + sample["options"][key] + ' \n'

        def takeSecond(elem):
            return elem[1]

        # sort list with key
        scores.sort(key=takeSecond, reverse=True)
        predicted_id = scores[0][0]
        predictions.append(sample["options"][predicted_id])


    return predictions


def FS_pred(train_splits, test_splits, model_name,prompt_size=5):
    # results_file = "FewShot_" + name + "_" + str(prompt_size) + ".csv"
    model_class, tokenizer_class = get_model_and_tokenizer(model_name)
    model = model_class.from_pretrained(model_name).to('cuda')
    tokenizer = tokenizer_class.from_pretrained(model_name)
    all_preds = []

    prompt = ''
    # prompt_size = 5
    # generate prompt sample
    for index in range(prompt_size):
        p = few_shot_prompt(train_splits[index], True)
        prompt += p

        # print("Prompt:")
        # print(prompt)
    predictions = fewshot(test_splits, model, tokenizer, normalize=True, prompt=prompt)

    return predictions