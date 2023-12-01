"""
This module is to evaluate the writing quality of the test data fetched from wandb.
"""

import sys
import re
import json
from io import StringIO
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import wandb
import logging
from tqdm.auto import tqdm
import openai
from CtrlEval.ctrl_evaluation import *


wandb.login(key='3138e1b24deb278ed045d0dedb39511d3a96245b')


# set up openai api key
openai.api_key = "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"

# set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='the name of the model to load from HuggingFace')
parser.add_argument('--run_name', type=str, help='the name of run to fetch the test data from wandb')
# below are arguments for logging purpose
parser.add_argument('-k', type=int, default=1, help='the number of few-shot examples')
parser.add_argument('--dataset', type=str, default='samsum', help='the dataset to evaluate on')
parser.add_argument('--keywords', type=str, default=None, choices=['tfidf'], help='the types of keywords to use')
parser.add_argument('--keyword_num', type=int, default=None, help='the number of keywords to use')
parser.add_argument('--control', type=str, default=None, choices=['length', 'entity', 'focus'],
                    help='the type of control')
parser.add_argument('--sample_num', type=int, default=-1, help='the number of samples to evaluate')
parser.add_argument('--mode', type=str, default='coherence', choices=['coherence', 'relevance'],
                    help='the mode of evaluation')

# parse the arguments
args = parser.parse_args()

wandb.init(
        project="In-context-learning for Dialogue Summarization",
        # track hyperparameters and run metadata
        config={
                'model_type': args.model,
                'k': args.k,
                'dataset': args.dataset,
                'keywords': args.keywords,
                'keyword_num': args.keyword_num,
                'run_name': args.run_name,
                },
        group='performance_in_context_learning',
        job_type='writing_evaluation'
        )

# set up log files
logging.basicConfig(
        level=logging.INFO,  # otherwise huggingface has many debug logs
        handlers=[
                logging.FileHandler("{}.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))),
                logging.StreamHandler(sys.stdout)
                ]
        )

# connect to wandb and get list of runs
api_helper = wandb.Api(api_key='3138e1b24deb278ed045d0dedb39511d3a96245b')
runs = list(api_helper.runs(path='yuting_fyp/In-context-learning for Dialogue Summarization',
                            per_page=1000))
logging.info(f'Number of runs fetched from WanDB: {len(runs)}')
logging.info(f'The evaluation mode is {args.mode}')


# iterate over runs to get the artifects
dfs = []
for run in tqdm(runs):
    if 'complete' not in run.tags or getattr(run, 'job_type', '') != 'evaluation':
        continue
    if run.name != args.run_name:  # skip the irrelevant runs
        continue
    files = run.files()
    metric_file = [f for f in files if 'Summaries Table' in getattr(f, 'name', '')]
    assert len(metric_file) == 1
    metric_file = metric_file[0]
    f = metric_file.download(root='wandb', replace=True)
    summaries = json.load(f)
    summaries = json.dumps(summaries)
    df = pd.read_json(StringIO(summaries), orient='split')
    dfs.append(df)
assert len(dfs) >= 1
logging.info(f'Number of tables fetched from WanDB: {len(dfs)}')

df = pd.concat(dfs, axis=0)
logging.info(f'shape of the dataframe: {df.shape}')


# group the df by run_id, and iterate the group to get average perplexity
grouped = df.groupby('run_id')
perplexities = []
gpt_response_df = []
for run_id, group_df in tqdm(grouped):
    if args.sample_num > 0:
        group_df = group_df.iloc[:args.sample_num]
        logging.info('Evaluate only on {} samples'.format(args.sample_num))
    # iterate the rows of group_df
    for _, row in tqdm(group_df.iterrows()):
        prompt_text = row['prompt']
        pred_summary = row['pred_summary']
        # ref: https://huggingface.co/docs/transformers/perplexity
        try:
            # parse the dialogue from the prompt_text
            dialogue = re.split(r'Summarize the conversation(.+)?:',
                                prompt_text)[-1].strip()
            # remove the line in dialogue start with 'Summary'
            dialogue = '\n'.join([line for line in dialogue.split('\n') if not line.startswith('Summary')])

            # call the evaluation function
            results = writing_evaluation(dialogue, pred_summary, mode=args.mode)

            # append row to gpt_response_df
            gpt_response_df.append({'run_id': run_id,
                                    'prompt': prompt_text,
                                    'pred_summary': pred_summary,
                                    'gpt_response': results['response_text'],
                                    args.mode: results[args.mode],
                                    })
        except Exception as e:
            logging.info('[error] {}'.format(e))
            logging.info('[error] prompt_text: {}'.format(prompt_text))
            logging.info('[error] pred_summary: {}'.format(pred_summary))
            # append row to gpt_response_df
            gpt_response_df.append({'run_id': run_id,
                                    'prompt': prompt_text,
                                    'pred_summary': pred_summary,
                                    'gpt_response': None,
                                    'coherence': None,
                                    'relevance': None,
                                    })

    break  # only evaluate the first run

# convert gpt_response_df to df
gpt_response_df = pd.DataFrame(gpt_response_df)
gpt_response_df = wandb.Table(dataframe=gpt_response_df)
wandb.log({"GPT response Table": gpt_response_df})
wandb.finish()
