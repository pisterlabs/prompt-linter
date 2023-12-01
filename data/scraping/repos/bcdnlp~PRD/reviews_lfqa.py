import json
import os
import re
from collections import OrderedDict

import pandas as pd
import yaml
from html2text import html2text
from tqdm import tqdm

with open('config.yml') as f:
    config = yaml.safe_load(f)

import anthropic_api
import openai_api
from prompt_generation import prompt_review_generation_lfqa

def result_extraction(text):
    try:
        preference = int(text.strip().split('\n')[-1].strip())
        return preference
    except:
        return -1

def run(reviewer_no):
    reviewer = config[f'reviewer{reviewer_no}']
    print(reviewer)

    data_path = config['lfqa_path']
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]

    # filter out duplicated questions in LFQA, only store one instance for each question
    new_data = OrderedDict()
    for d in data:
        if d['question'] not in new_data:
            new_data[d['question']] = d
    data = list(new_data.values())
    
    reviews = []
    for d in tqdm(data):
        # extract information
        question = html2text(d['question']).strip()
        answer_a = html2text(d['answer_a']).strip()
        answer_b = html2text(d['answer_b']).strip()
        answer_a_type = d['answer_a_type']
        answer_b_type = d['answer_b_type']
        preference_human = int(d['overall_preference'] / 2 + 3 / 2) # -1, 1 to 1, 2
        justification_human = d['justification']

        # generate review prompts
        sys_prompt, prompt = prompt_review_generation_lfqa(question, answer_a, answer_b)
        history = [sys_prompt, prompt]

        # call apis to generate reviews
        if 'claude' == reviewer[:6]:
            justification_reviewer = anthropic_api.call(history, reviewer_no)
        elif 'gpt' == reviewer[:3]:
            justification_reviewer = openai_api.call(history, reviewer_no)
        # automatic extraction
        preference_reviewer = result_extraction(justification_reviewer)

        # store results
        reviews.append({'question': question,
                        'answer_a': answer_a,
                        'answer_b': answer_b,
                        'answer_a_type': answer_a_type,
                        'answer_b_type': answer_b_type,
                        f'justification_{reviewer}': justification_reviewer,
                        'justification_human': justification_human,
                        f'preference_{reviewer}': preference_reviewer,
                        'preference_human': preference_human,
                        })

    # save results
    save_path = os.path.join(config['lfqa_results_folder'],
                             config['lfqa_review_file'].format(
                                reviewer=config[f'reviewer{reviewer_no}'],
                                temperature=config[f'temperature_reviewer{reviewer_no}'],
                                mode=config['mode']))
    print(save_path)
    with open(save_path, 'w') as f:
        json.dump(reviews, f, indent=2)

if __name__ == '__main__':
    # generate reviews for both reviewers
    run(1)
    run(2)

