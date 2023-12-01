import openai
import json
from transformers import GPT2Tokenizer
import math
import os
import numpy as np

from gpt3.laria.gpt_util import logprobs_to_probs, tokenize

openai.api_key = = os.environ["OPENAI_API_KEY"]

# TODO save metadata and make readable
def generate_subtree(seed, branching_factor, branching_interval, depth,
                     engine='ada', temperature=0.7, sampling='stochastic', root=True):
    if depth == 0:
        return 'leaf'
    tree = {}
    response = openai.Completion.create(
        engine=engine,
        prompt=seed,
        temperature=temperature,
        max_tokens=branching_interval,
        echo=False,
        top_p=1,
        n=branching_factor,
        logprobs=0
    )
    tree['response'] = response
    tree['children'] = []
    for i in range(branching_factor):
        subtree = generate_subtree(seed + response.choices[i]['text'],
                                   branching_factor,
                                   branching_interval,
                                   depth - 1,
                                   engine,
                                   temperature,
                                   sampling,
                                   root=False)
        if not subtree == 'leaf':
            tree['children'].append(subtree)
    return tree


# warning: consumes many tokens!
# at each step, branches until cumulative probability of branches meets probability threshold
def adaptive_subtree(seed, init_text=None, init_prob=None, depth=5, probability_threshold=.15,
                     engine='ada', temperature=0, sampling='stochastic', root=True):
    if depth == 0:
        return 'leaf'
    tree = {}
    if init_text is not None:
        tree['text'] = init_text
    else:
        tree['text'] = seed
    tree['children'] = []
    tree['probability'] = init_prob
    cumulative_prob = 0
    branches = []
    probs = []
    mask = {}
    while cumulative_prob < probability_threshold:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=seed,
                temperature=temperature,
                max_tokens=1,
                echo=False,
                top_p=1,
                n=1,
                logprobs=0,
                logit_bias=mask
                )
        except Exception as e:
            print(e)
            return 'leaf'
        token = response.choices[0]["logprobs"]["tokens"][0]
        try:
            token_id = tokenize([token])[0][0]
        except Exception as e:
            print(e)
            print(token)
            return 'leaf'
        mask[token_id] = -100
        branches.append(token)
        logprob = response.choices[0]["logprobs"]["token_logprobs"][0]
        prob = logprobs_to_probs([logprob])[0]
        probs.append(prob)
        cumulative_prob += prob

    for i, branch in enumerate(branches):
        subtree = adaptive_subtree(seed=seed + branch,
                                   init_text=branch,
                                   init_prob=probs[i],
                                   depth=depth-1,
                                   probability_threshold=probability_threshold,
                                   engine=engine,
                                   temperature=temperature)
        if not subtree == 'leaf':
            tree['children'].append(subtree)
    return tree

# creates n branches and splits each one by least confident token
def adaptive_branch(seed, branching_factor=3, branching_interval=20,
                     engine='ada', temperature=0.7, sampling='stochastic'):
    tree = {}
    tree['text'] = seed
    response = openai.Completion.create(
        engine=engine,
        prompt=seed,
        temperature=temperature,
        max_tokens=branching_interval,
        echo=False,
        top_p=1,
        n=branching_factor,
        logprobs=0
    )
    tree['children'] = []
    for choice in response.choices:
        min_logprob = np.argmin(choice["logprobs"]["token_logprobs"])
        split_position = choice["logprobs"]["text_offset"][min_logprob]-len(prompt)
        # print(min_logprob)
        # print(split_position)
        childtext = choice["text"][:split_position]
        grandchild_text = choice["text"][split_position:]
        grandchild = {'text': grandchild_text, 'children': []}
        tree['children'].append({'text': childtext,
                                 'children': [grandchild]})
        print(childtext)
        print(grandchild_text)
        for i, token in enumerate(choice["logprobs"]["tokens"]):
            print(token, choice["logprobs"]["token_logprobs"][i])

    return tree


def print_tree(tree):
    for i, child in enumerate(tree['children']):
        print(tree['response'].choices[i]['text'])
        print_tree(child)


def make_readable_tree(tree):
    readable_tree = []
    for response in tree['response'].choices:
        readable_tree.append({'text': response['text']})
    for i, child in enumerate(tree['children']):
        readable_tree[i]['children'] = make_readable_tree(child)
    return readable_tree


f = open("prompt.txt", "r")
prompt = f.read()
print(prompt)
# tree = generate_subtree(prompt, branching_factor=3,
#                         branching_interval=10,
#                         depth=5,
#                         temperature=0.9,
#                         engine='ada')
# print_tree(tree)
# # readable_tree = make_readable_tree(tree)
#  tree_json = {'text': prompt,
#               'children': readable_tree}

tree = adaptive_branch(prompt)

with open('../../data/adaptive_ada_branch.json', 'w') as outfile:
    json.dump(tree, outfile)
