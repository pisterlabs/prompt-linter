from typing import List
from tqdm import tqdm
import openai
from multiprocessing import Pool
from functools import partial
import math
import argparse
import os
import shutil
import json
import time
import numpy as np
import math
import warnings
import backoff

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input", help="input task file or dir containing task files")
parser.add_argument('-o', "--output", default="gpt3-results", help="output dir")
parser.add_argument("--template", default="paraphrase.prompt", help="template file")
parser.add_argument('-m', "--max_tries", type=int, default=4, help="number of paraphrases to generate")
parser.add_argument("--num_workers", type=int, default=5, help="number of processes used during generation")

# GPT-3 generation hyperparameters
parser.add_argument('--engine', type=str, default='text-davinci-002',
                    choices=['ada',
                             'text-ada-001',
                             'babbage',
                             'text-babbage-001',
                             'curie',
                             'text-curie-001',
                             'davinci',
                             'text-davinci-001',
                             'text-davinci-002'],
                    help='The GPT-3 engine to use.')  # choices are from the smallest to the largest model
parser.add_argument('-t', '--max_tasks', type=int, default=None, help='Maximum number of tasks to process, if provided and\
                                                            smaller than input folder, random sampling is performed. The \
                                                            selected tasks are copied to the "selected_tasks" folder.')
parser.add_argument('--max_tokens', type=int, default=40, required=False, help='')
parser.add_argument('--temperature', type=float, default=0.8, required=False, help='')
parser.add_argument('--top_p', type=float, default=0.9, required=False, help='')
parser.add_argument('--frequency_penalty', type=float, default=0.0, required=False, help='')
parser.add_argument('--presence_penalty', type=float, default=0.0, required=False, help='')
parser.add_argument('--stop_tokens', nargs='+', type=str,
                        default=None, required=False, help='Stop tokens for generation')
args = parser.parse_args()
MAXIMUM_TRIES_BEFORE_GIVE_UP = args.max_tries * 2

def fill_template(prompt_template_file: str, **prompt_parameter_values) -> str:
    prompt = ''
    with open(prompt_template_file) as prompt_template_file:
        for line in prompt_template_file:
            if line.startswith('#'):
                continue  # ignore comment lines in the template
            prompt += line
    for parameter, value in prompt_parameter_values.items():
        prompt = prompt.replace('{'+parameter+'}', value)
    return prompt

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def generate_one(input_text: str, args) -> str:
    generation_output = openai.Completion.create(engine=args.engine,
                                                 prompt=input_text,
                                                 max_tokens=max(args.max_tokens, len(input_text.split(' '))),
                                                 temperature=args.temperature,
                                                 top_p=args.top_p,
                                                 frequency_penalty=args.frequency_penalty,
                                                 presence_penalty=args.presence_penalty,
                                                 best_of=1,
                                                 stop=args.stop_tokens,
                                                 logprobs=0,  # log probability of top tokens
                                                 )
    # print('raw generation output = ', generation_output)
    # print('='*10)
    generation_output = generation_output['choices'][0]['text']
    generation_output = generation_output.strip()
    return generation_output

def generate(input_text: str, args, existing=[], postprocess=True) -> str:
    """
    text-in-text-out interface to large OpenAI models
    """
    # don't try multiple times if the temperature is 0, because the results will be the same
    if args.max_tries > 1:
        assert args.temperature != 0
    # try until get max_tries useful prompts
    outputs = set(existing)
    if len(outputs) > 0:
        args.temperature = 0.9 # increase temperature to increase chance of success
    while True:
        if len(outputs) >= args.max_tries:
            break
        generation_output = generate_one(input_text, args)
        if postprocess:
            generation_output = _postprocess_generations(generation_output)
        if len(generation_output) > 0:
            outputs.add(generation_output)
    # if len(outputs) < args.max_tries:
    #     warnings.warn("Not enough outputs are generated!", DeprecationWarning)
    return list(outputs)

# def generate_AI21(input_text: str, args, postprocess=True, max_tries=1) -> str:
#     """
#     text-in-text-out interface to large AI21 models
#     """
    
#     # don't try multiple times if the temperature is 0, because the results will be the same
#     if max_tries > 1 and args.temperature == 0:
#         max_tries = 1
#     # try at most `max_tries` times to get a non-empty output
#     for _ in range(max_tries):
        
#         response = requests.post(
#             "https://api.ai21.com/studio/v1/experimental/rewrite",
#             headers={"Authorization": "Bearer [API KEY]"},
#             json={
#                 "text": input_text,
#                 "intent": "long"
#             }
#         )
        
#         # print('raw generation output = ', generation_output)
#         # print('='*10)
#         # print(response.json())
#         generation_output = [gen for gen in response.json()['suggestions']['text']]
#         if len(generation_output) < 1:
            
#         if postprocess:
#             generation_output = _postprocess_generations(
#                 generation_output)
#         if len(generation_output) > 0:
#             break
#     return generation_output

def batch_generate(input_texts: List[str], args, postprocess=True, num_processes=5) -> List[str]:
    """
    Call OpenAI's API in parallel, since each call to the biggest model takes ~1 second to return results
    """
    f = partial(generate, args=args, postprocess=postprocess)
    with Pool(num_processes) as p:
        worker_outputs = list(
            tqdm(p.imap(f, input_texts), total=len(input_texts)))
    return worker_outputs

def _postprocess_generations(generation_output: str) -> str:
    """
    Might output an empty string if generation is not at least one full sentence
    """
    # replace all whitespaces with a single space
    generation_output = ' '.join(generation_output.split())

    # delete half sentences
    generation_output = generation_output.strip()
    if len(generation_output) == 0:
        return generation_output

    if generation_output[-1] not in {'.', '!', '?'}:
        last_sentence_end = max(generation_output.find('.'), generation_output.find('!'), generation_output.find('?'))
        if last_sentence_end > 0:
            generation_output = generation_output[:last_sentence_end+1]

    # remove extraneously generated outputs
    if 'Original:' in generation_output and 'Paraphrase:' in generation_output:
        generation_output = generation_output[:generation_output.find('Original:')]

    return generation_output


if not os.path.exists(args.output):
    os.makedirs(args.output)

task_paths = []
if os.path.isdir(args.input):
    task_paths = [os.path.join(args.input, filename) for filename in os.listdir(args.input)]
else:
    task_paths = [args.input]

if args.max_tasks is not None and args.max_tasks < len(task_paths):
    rng = np.random.default_rng()
    task_paths = rng.choice(task_paths, size=args.max_tasks, replace=False).tolist()
    if not os.path.exists('selected_tasks'):
        os.makedirs('selected_tasks')
    for task_path in task_paths:
        folder, file = os.path.split(task_path)
        folder, par = os.path.split(folder)
        dst = os.path.join(folder, 'selected_tasks', file)
        shutil.copy(task_path, dst)

task_names = []
orig_prompts = [] # the Definition in the original task
prompts = [] # templated prompt for gpt3
for task_file in task_paths:
    task_dict = json.load(open(task_file, 'r'))
    instruction = "\n".join(task_dict['Definition'])
    if not instruction.endswith('.'):
        instruction += '.' # prevent gpt3 generating a dot
    orig_prompts.append(instruction)
    task_name = os.path.basename(task_file)
    task_names.append(task_name)
    prompt = fill_template(args.template, instruction=instruction)
    prompts.append(prompt)

#results = batch_generate(prompts, args, num_processes=args.num_workers)
for i, prompt in enumerate(tqdm(prompts)):
    try:
        task_name = task_names[i]
        orig_prompt = orig_prompts[i]
        save_file = os.path.join(args.output, args.engine+'_'+args.template+'_'+task_name)
        
        existing = []
        if os.path.exists(save_file):
            existing = json.load(open(save_file, 'r'))['generated_prompts']
            if len(existing) >= args.max_tries:
                continue
        
        gen_prompts = generate(prompt, args, existing=existing)
        
        data_dict = {
            'orignal_task': task_name,
            'engine': args.engine,
            'template': args.template,
            'original_prompt': orig_prompt,
            'generated_prompts': gen_prompts
        }
        json.dump(data_dict, open(save_file, 'w'), indent=4)
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(e)
        continue

# results = []
# for task_name, orig_prompt, gen_prompts in zip(task_names, orig_prompts, results):
#     # TODO: could also save the args
#     data_dict = {
#         'orignal_task': task_name,
#         'engine': args.engine,
#         'template': args.template,
#         'original_prompt': orig_prompt,
#         'generated_prompts': gen_prompts
#     }
#     save_file = os.path.join(args.output, args.engine+'_'+args.template+'_'+task_name)
#     json.dump(data_dict, open(save_file, 'w'), indent=4)