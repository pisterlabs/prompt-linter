import json
import os
import argparse
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import openai
import datasets
import sys
sys.path.append('../llm_zero_generate')
from get_response_llm import get_openai_api
from rankgpt_prompter import Prompter
import copy
import time

# def get_openai_api(model_name):
#     if model_name == "Qwen":
#         print("Using Qwen")
#         openai.api_base = "http://0.0.0.0:8111/v1"
#         openai.api_key = "xxx"
#     elif model_name == "Llama":
#         print("Using Llama")
#         openai.api_base = "http://0.0.0.0:8223/v1"
#         openai.api_key = "xxx"
#     elif model_name == "chatglm3":
#         print("Using chatglm3")
#         openai.api_base = "http://0.0.0.0:8113/v1"
#         openai.api_key = "xxx"
#     elif model_name == "baichuan2-13b-chat":
#         print("Using baichuan2-13b-chat")
#         openai.api_base = "http://0.0.0.0:8222/v1"
#         openai.api_key = "xxx"
#     elif model_name == "gpt-3.5-turbo":
#         print("Using gpt-3.5-turbo")
#         openai.api_base = "http://47.245.109.131:5555/v1"
#         openai.api_key = "sk-Y5UPxxoh10M9h50RBe8e70EfEc484556A80a8717623aEb2f"
#     else:
#         raise ValueError("Model name not supported")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=10))
def get_response_llm(model_name, message):
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user",
             "content": message},
        ],
        max_tokens=128,
        temperature=0,
        top_p=1,
        stream=False,
    )


    # print(text)
    # print(completion.choices[0].message.content)
    # print("\n")

    resp = completion.choices[0].message.content.strip().replace("\n", " ")
    # memory cleanup
    del completion
    return resp

def create_permutation_instruction(item=None, rank_start=0, rank_end=100, prompter=None, prf_num=0):
    query = item['query']
    hits = item['hits'][rank_start: rank_end]
    num = len(hits)

    passages = [{'rank': rank + 1, 'content': hit['content']} for rank, hit in enumerate(hits)]
    input_text = '\n'.join([f"[{p['rank']}] {p['content']}" for p in passages])
    if prf_num > 0:
        prf_hits = item['hits'][:prf_num]

        prf_passages = [{'rank': rank + 1, 'content': hit['content']} for rank, hit in enumerate(prf_hits)]
        feedback_documents = '\n'.join([f"prf{p['rank']}: {p['content']}" for p in prf_passages])

        full_prompt = prompter.generate_prompt(query, input_text, num=num, feedback_documents=feedback_documents)
    else:
        full_prompt = prompter.generate_prompt(query, input_text, num=num)

    return full_prompt


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item


def permutation_pipeline(item=None, rank_start=0, rank_end=100, model=None,  prompter=None,
                         prf_num=0):
    # count time for each step

    time_start = time.time()
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end, prompter=prompter,
                                              prf_num=prf_num)
    # messages = create_wtf_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end, prompter=prompter, tokenizer=tokenizer)
    time_end = time.time()
    print(f'create_permutation_instruction takes {time_end - time_start} seconds')
    print(f'messages is {messages}')
    time_start = time.time()
    # permutation = run_llm(messages, model=model, tokenizer=tokenizer, generate_config=generate_config, device=device, prompter=prompter)
    permutation = get_response_llm(model_name=model, message=messages)
    print(f'permutation is {permutation}')
    time_end = time.time()
    print(f'run_llm takes {time_end - time_start} seconds')
    time_start = time.time()
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    # print(f'item is {item}')
    time_end = time.time()
    print(f'receive_permutation takes {time_end - time_start} seconds')
    return item


def sliding_windows(item=None, rank_start=0, rank_end=100, window_size=20, step=10,
                    model=None, prompter_name='LLMreranker', prf_num=0):
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    prompter = Prompter(template_name=prompter_name)
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        print(f'start_pos is {start_pos}, end_pos is {end_pos}')
        # print(f'item is {item}')
        item = permutation_pipeline(item, start_pos, end_pos,
                                    model=model,
                                    prompter=prompter, prf_num=prf_num)
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item




# import guidance
#
#
# def get_guidance_llm(model_name):
#     guidance.llm = guidance.llms.OpenAI(model_name)
#     return guidance.llm
#
#
# @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=10))
# def get_gpt_result(model, message):
#     # {{gen 'strength' pattern = '[0-9]+' stop = ','}}
#     # generate formatted pattern strings like [num of rank_num]>[num of rank_num]..., stop by \n,
#
#     # select_str = '\[[0-9]+\]'
#     # for i in range(rank_num - 1):
#     #     pattern = '\>\[[0-9]+\]'
#     #     select_str += pattern
#     #
#     # select_str = """{{gen 'resp' pattern = '"""+select_str+"""' stop = '\\n'}}"""
#     # print(f'select_str is {select_str}')
#     # full_message =  message + """{{gen 'resp' temperature=0.7}}"""  #+ select_str
#     # print(f'full_message is {full_message}')
#     resp = guidance("""
#     {{#system~}}
#     You are a helpful and terse assistant.
#     {{~/system}}
#
#     {{#user~}}
#     """ + message + """
#     {{~/user}}
#
#     {{#assistant~}}
#     {{gen 'resp' temperature=0.7}}
#     {{~/assistant}}""", llm=model)()
#
#     print(f'resp is {resp["resp"]}')
#     return resp["resp"]