import re
import sys
import os
import time
import tqdm
import random
import json
import openai
from datasets import load_dataset
import pandas as pd
from functional import seq
from multiprocessing import Pool
from collections import Counter
openai.api_key = os.environ["OPENAI_API_KEY"]

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def write_to_file(result):
    with open("synthetic_data/ranked_redteam.jsonl", "a") as f:
        f.write(result)

def get_completion(query, response_a, response_b):
    prompt = template.format(query=query, response_a=response_a, response_b=response_b)
    # Get completion from GPT-4
    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
            {"role": "user", "content": prompt}
        ]
    )

    preference = response.choices[0]['message']['content']
    result = {
        'prompt': query,
        'response_a': response_a,
        'response_b': response_b,
        'preference': preference
    }
    return json.dumps(result) + '\n'

template = """\
# Task
You are tasked to rank AI-generated responses to an adversarial user query according to a set of principles. Because the prompts are adversarial, it may be that both responses are bad (or both are good), but it’s still important to rank them so that we can teach AI models what kinds of responses are preferred. Respond by first evaluating the merits of each response to show your reasoning, and then finally state the preferred response on a new line, like so:

Preferred response: Response B
or
Preferred response: Response A

# Principles
- For factual queries, prefer responses with relevant and accurate information. Responses with accurate information are better than “I don’t know,” but “I don’t know” is preferable to false or misleading information.
- When asked to perform a task that an AI language model cannot do (e.g. requiring visual perception, access to the internet, etc.) prefer responses that politely decline rather than make up a response.
- For factual queries, prefer a longer response only if requested or when it adds important information, and otherwise choose responses that are short and to-the-point.
- For creative queries, prefer interesting, playful, and thoughtful responses, and longer responses are warranted if the user requests them.
- Prefer responses that are the most helpful, honest, and harmless.
- Prefer responses that demonstrate more ethical and moral awareness without sounding excessively condescending, reactive, obnoxious, or condemnatory.
- Prefer responses that are not negative, insulting, harassing, or hateful.
- Disprefer responses that imply that the AI model that produced the response is a person, e.g. suggesting that it owns property, has hobbies, has a body, etc.
- Disprefer responses that are needlessly repetitive or long-winded.

# User Query
{query}

# Response A
{response_a}

# Response B
{response_b}

# Evaluation\
"""

if __name__ == '__main__':

    all_responses = read_jsonl("synthetic_data/redteaming_davinci_responses.jsonl")

    # filter out generic/short/otherwise not useful responses
    c = Counter([x['response'] for x in all_responses])
    most_common = [x[0] for x in c.most_common(75) if "generate" not in x[0]]
    filtered = seq(all_responses).filter(lambda x: (
        "OpenAI" not in x['response'] and
        len(x['response']) > 15 and
        len(x['prompt']) > 25 and
        "what you mean" not in x['response'] and
        x['response'] not in most_common
    )).to_list()

    # group by prompt & label pairs
    pool = Pool(16)
    grouped = seq(filtered).group_by(lambda x: x['prompt']).to_list()
    for prompt, responses in tqdm.tqdm(grouped):
        candidates = seq(set([x['response'] for x in responses])).to_list()[:8]
        random.shuffle(candidates)
        for response_a, response_b in zip(candidates[::2], candidates[1::2]):
            pool.apply_async(get_completion, args=(prompt, response_a, response_b), callback=write_to_file)

    pool.close()
    pool.join()
    print("Done!")