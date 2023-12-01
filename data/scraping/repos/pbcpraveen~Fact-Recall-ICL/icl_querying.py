import os
import json
import requests
import argparse
import openai
from dotenv import load_dotenv
from tqdm import tqdm
from nltk import word_tokenize
import string
import pdb
from tqdm import tqdm
import statistics
import pandas as pd
import numpy as np
from datasets import load_dataset
load_dotenv('api_key.env')
import re
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from utils import *
import random
import threading
import time
openai.api_key = os.getenv("OPENAI_API_KEY")




COUNT = 500
n_threads = 5

responses = [[] for _ in range(n_threads)]
responses_indirect = [[] for _ in range(n_threads)]

def querying_indirect_thread(prompts, index):
    global responses
    c = len(prompts)
    i = 0
    responses_thread = []
    pbar = tqdm(total=c)
    while (len(responses_thread) != c):
        try:
            query = [
                {"role": "user", "content": prompts[i]}
            ]
            response = chatgpt_query(query)
            i += 1
            responses_thread.append(response)
            pbar.update(1)
        except:
            print('sleeping')
            time.sleep(10)
    pbar.close()
    responses_indirect[index] = responses_thread

def querying_thread(prompts, index):
    global responses
    c = len(prompts)
    i = 0
    responses_thread = []
    pbar = tqdm(total=c)
    while (len(responses_thread) != c):
        try:
            query = [
                {"role": "user", "content": prompts[i]}
            ]
            response = chatgpt_query(query)
            i += 1
            responses_thread.append(response)
            pbar.update(1)
        except:
            print('sleeping')
            time.sleep(10)
    pbar.close()
    responses[index] = responses_thread

df = pd.read_csv("result/sample1_prompt1_response.csv")
response_year = df.apply(lambda x: extract_year(x['GPT 4 Response Prompt1']), axis=1)
df['response_year'] = response_year
filtered = df[[df['response_year'].iloc[:].tolist()[i] != None for i in  range(1000)]]
correctly_answered =  filtered[filtered.apply(check_year, axis=1)]
examples = correctly_answered.apply(generate_example1, axis = 1)
correctly_answered['example'] = examples
correctly_answered['example_indirect'] = correctly_answered.apply(generate_example_statement_indirect, axis = 1)

example_count = COUNT
icl_prompt_indirect = []
icl_prompt = []
ground_truth = []
citations_query = []
mean_citation_example = []
i = 0
visited = set()
while i < example_count:
    sample = correctly_answered.sample(4)
    query = sample.iloc[3]['name']
    if query in visited:
        continue
    visited.add(query)
    icl_prompt_indirect.append(generate_icl_query_indirect(sample))
    icl_prompt.append(generate_icl_query(sample))
    ground_truth.append(sample.iloc[3]['wikipedia_birth_year'])
    citations_query.append(sample.iloc[3]['citation'])
    mean_citation_example.append(sum([sample.iloc[i]['citation'] for i in range(3)])/3)
    i+=1
prompts = icl_prompt
prompts_indirect = icl_prompt_indirect

df = pd.DataFrame()
df['prompt'] = prompts
df['prompt_indirect'] = icl_prompt_indirect
df['ground_truth'] = ground_truth
df['citations_query'] = citations_query
df['mean_citation_example'] = mean_citation_example

partitions = []
partitions_indirect = []
bin_size = COUNT // n_threads
for i in range(n_threads - 1):
    partitions.append(prompts[i * bin_size: (i+1) * bin_size])
    partitions_indirect.append(prompts_indirect[i * bin_size: (i+1) * bin_size])
partitions.append(prompts[(n_threads - 1) * bin_size:])
partitions_indirect.append(prompts_indirect[(n_threads - 1) * bin_size:])
threads = []
for i in range(n_threads):
    threads.append(threading.Thread(target=querying_thread, args=(partitions[i], i,)))


print("starting API resquests to OPENAI's GPT 4 using ", n_threads, " threads")
print("Number of threads created: ", len(threads))
print("Number of partitions created: ", len(partitions))

for i in range(n_threads):
    threads[i].start()
for i in range(n_threads):
    threads[i].join()
                      
threads = []
for i in range(n_threads):
    threads.append(threading.Thread(target=querying_indirect_thread, args=(partitions[i], i,)))


print("starting API resquests to OPENAI's GPT 4 using ", n_threads, " threads")
print("Number of threads created: ", len(threads))
print("Number of partitions created: ", len(partitions))

for i in range(n_threads):
    threads[i].start()
for i in range(n_threads):
    threads[i].join()

responses = list(itertools.chain(*responses))
responses_indirect = list(itertools.chain(*responses_indirect))

df['GPT 4 Response'] = responses
                      
df['GPT 4 Response Indirect'] = responses_indirect                  

df.to_csv("result/sample1_icl_indirect_prompt_response.csv")

