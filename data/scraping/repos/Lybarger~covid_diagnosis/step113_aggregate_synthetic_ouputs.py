




import os
import openai
from openai.embeddings_utils import cosine_similarity, get_embedding

import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import jsonlines
import logging
import aiohttp
import asyncio  # for running API calls concurrently
import random
import argparse
from collections import Counter, OrderedDict
import string

import paths
import config
from utils import make_and_clear

import api_request_parallel_processor_new as api_request
from api_request_parallel_processor_new import CHOICES, MESSAGE, CONTENT, USAGE, PROMPT_TOKENS, COMPLETION_TOKENS, TOTAL_TOKENS, FINISH_REASON, MESSAGES

from config import CHAT_RPM_LIMIT, CHAT_TPM_LIMIT, CHAT_URL




    
def char_check(x):
    x=x.decode('utf-8','ignore').encode("utf-8")
    return x

def remove_non_ascii(a_str):
    ascii_chars = set(string.printable)

    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    )

def main(args):

    destination = args.destination
    make_and_clear(destination)


    total_tokens_all = 0
    finish_reason_all = Counter()
    messages_all = []
    prompts_all = []

    with open(args.synthetic_responses, 'r') as file:
        X = file.__iter__()
        for x in X:
            input, output = tuple(eval(x))

            prompt = input[MESSAGES][0][CONTENT]
            prompt = remove_non_ascii(prompt)
            prompts_all.append(prompt)

            prompt_tokens = output[USAGE][PROMPT_TOKENS]
            completion_tokens = output[USAGE][COMPLETION_TOKENS]
            total_tokens = output[USAGE][TOTAL_TOKENS]
            total_tokens_all += total_tokens

            finish_reason = output[CHOICES][0][FINISH_REASON]
            finish_reason_all[finish_reason] += 1


            message = output[CHOICES][0][MESSAGE][CONTENT]
            message = message.lstrip()
            message = remove_non_ascii(message)

            messages_all.append(message)




    average_tokens = float(total_tokens_all)/len(messages_all)

    total_cost = total_tokens_all*args.cost_per_token
    average_cost = average_tokens*args.cost_per_token

    print(f'Total tokens: {total_tokens_all}')
    print(f'Average tokens: {average_tokens}')   

    print(f'Cost, total:    {total_cost}')
    print(f'Cost, average:  {average_cost}')   

    print(f'Finish reason')
    for k, v in finish_reason_all.items():
        print(f'\t{k} = {v}')


    df = pd.DataFrame(zip(prompts_all, messages_all), columns=['prompts', 'completions'])
    f = os.path.join(destination, 'prompt_completion_pairs.csv')
    df.to_csv(f)


    prompt_completion_dict = OrderedDict(zip(prompts_all, messages_all))
    def get_completion(x, d=prompt_completion_dict):
        return d.get(x, 'None')
    
    df = pd.read_csv(args.synthetic_prompts)
    df['Synthetic_prompt'] = df[config.PROMPT].apply(get_completion)
    f = os.path.join(destination, 'regression_spreadsheet.csv')
    df.to_csv(f)
    

if __name__ == '__main__':


    destination = paths.aggregate_synthetic_ouputs
    synthetic_prompts = paths.synthetic_prompts_async
    synthetic_responses = paths.synthetic_responses_async


    cost_per_1000_tokens = 0.002
    cost_per_token = cost_per_1000_tokens/1000

    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('--destination', type=str, default=destination, help="output directory")
    arg_parser.add_argument('--synthetic_prompts', type=str, default=synthetic_prompts, help="output directory")    
    arg_parser.add_argument('--synthetic_responses', type=str, default=synthetic_responses, help="output directory")    
    arg_parser.add_argument('--cost_per_token', type=float, default=cost_per_token, help="output directory") 

    args, _ = arg_parser.parse_known_args()


    sys.exit(main(args)) 
