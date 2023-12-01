import json
import os
import openai
import random
import time
from datetime import datetime
import argparse
import re
from tqdm import tqdm
from typing import Union
from prompts import date_prompt
from collections import OrderedDict
from tool import *


def get_user_assistant_messages(system_message: str, user_message: str, assistant_message: str):
    '''
    This function is used to convert the prompt into the message format used by OpenAI Chat API.
    '''
    messages = []
    messages.append({"role": "system", "content": system_message})
    split_user_messages = user_message.split('\n\n\n\n')
    split_assistant_messages = assistant_message.split('\n\n\n\n')
    for i in range(len(split_user_messages)):
        question = split_user_messages[i]
        answer = split_assistant_messages[i]
        messages += [
            {"role": "user", "content": f"{question}"},
            {"role": "assistant", "content": f"{answer}"},
        ]
    return messages


def get_cot_prompt(data: dict):
    '''
    This function is used to generate the prompt for CoT.
    '''
    system_message = date_prompt.TURBO_COT_SYSTEM
    user_message = date_prompt.TURBO_COT_USER
    assistant_message = date_prompt.TURBO_COT_ASSISTANT

    messages = get_user_assistant_messages(
        system_message, user_message, assistant_message)
    question_message = data['question']
    messages += [{"role": "user", "content": f"Question: {question_message}"}]

    return messages


def get_pal_prompt(data: dict):
    '''
    This function is used to generate the prompt for PAL.
    '''
    system_message = date_prompt.TURBO_PAL_SYSTEM
    user_message = date_prompt.TURBO_PAL_USER
    assistant_message = date_prompt.TURBO_PAL_ASSISTANT

    messages = get_user_assistant_messages(
        system_message, user_message, assistant_message)

    question_message = data['question']
    messages += [{"role": "user",
                  "content": f"Question: {question_message}\n\n# solution in Python"}]

    return messages


def get_select_prompt(data: dict, cot_solution: list, pal_solution: list):
    '''
    This function is used to generate the prompt for selection.
    '''
    system_message = date_prompt.TURBO_SELECT_SYSTEM
    user_message = date_prompt.TURBO_SELECT_USER
    assistant_message = date_prompt.TURBO_SELECT_ASSISTANT
    messages = get_user_assistant_messages(
        system_message, user_message, assistant_message)

    try:
        pal_generated_list = pal_solution[0].split('"""')
        pal_generated = pal_generated_list[0].strip(
        ) + pal_generated_list[2]
    except Exception as e:
        pal_generated = pal_solution[0]

    if cot_solution[0].startswith('Answer:'):
        cot_generated = cot_solution[0]
    else:
        cot_generated = 'Answer:\n' + cot_solution[0]

    user_message = f'''Math problem: {data['question'].strip()}

(A)
{cot_generated.strip()}

(B)
{pal_generated.strip()}

Which of the above two choices can correctly answer the math problem?'''

    messages += [{"role": "user", "content": user_message}]

    return messages


def query_cot(data: dict, key: str, cot_temperature: float, backbone: str):
    '''
    This function is used to query OpenAI for CoT solutions.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        cot_temperature: the temperature used in CoT
        backbone: ChatGPT or GPT-4

    Returns:
        completions: a list containing the CoT solution
    '''
    query_message = get_cot_prompt(data)
    if backbone == 'gpt4':
        model_name = 'gpt-4'
    elif backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo'

    start_time = time.time()
    completions = []
    while True:
        try:
            cot_solution = openai.ChatCompletion.create(
                api_key=key,
                model=model_name,
                max_tokens=500,
                stop='\n\n\n',
                messages=query_message,
                temperature=cot_temperature,
                top_p=1.0,
                n=1)
        except Exception as e:
            cot_solution = None

        if cot_solution is not None:
            completions.extend([choice['message']['content']
                                for choice in cot_solution['choices']])
            completions = completions[:1]
            return completions
        else:
            sleep_time = random.uniform(3, 5)
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None


def query_pal(data: dict, key: str, pal_temperature: float, backbone: str):
    '''
    This function is used to query OpenAI for PAL solutions.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        pal_temperature: the temperature used in PAL
        backbone: ChatGPT or GPT-4

    Returns:
        completions: a list containing the PAL solution
    '''
    query_message = get_pal_prompt(data)
    if backbone == 'gpt4':
        model_name = 'gpt-4'
    elif backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo'
    start_time = time.time()
    completions = []
    while True:
        try:
            pal_solution = openai.ChatCompletion.create(
                api_key=key,
                model=model_name,
                max_tokens=500,
                stop='\n\n\n',
                messages=query_message,
                temperature=pal_temperature,
                top_p=1.0,
                n=1)
        except Exception as e:
            pal_solution = None

        if pal_solution is not None:
            completions.extend([choice['message']['content']
                                for choice in pal_solution['choices']])
            completions = completions[:1]
            return completions
        else:
            sleep_time = random.uniform(3, 5)
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None


def query_select(data: dict, key: str, cot_solution: list, pal_solution: list, backbone: str):
    '''
    This function is used to query OpenAI for selection solutions.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        cot_solution: a list containing the CoT solution
        pal_solution: a list containing the PAL solution
        backbone: ChatGPT or GPT-4

    Returns:
        completions: a list containing the selection solution
    '''
    selection_message = get_select_prompt(
        data, cot_solution, pal_solution)
    if backbone == 'gpt4':
        model_name = 'gpt-4'
    elif backbone == 'chatgpt':
        model_name = 'gpt-3.5-turbo'
    start_time = time.time()
    completions = []
    while True:
        try:
            selection_solution = openai.ChatCompletion.create(
                api_key=key,
                model=model_name,
                max_tokens=200,
                stop='\n\n',
                messages=selection_message,
                temperature=0.,
                top_p=1.0,
                n=1)
        except Exception as e:
            selection_solution = None

        if selection_solution is not None:
            completions.extend([choice['message']['content']
                                for choice in selection_solution['choices']])
            completions = completions[:1]
            return completions
        else:
            sleep_time = random.uniform(3, 5)
            time.sleep(sleep_time)

        if time.time() - start_time > 60:
            return None


def query_date(data: dict, key: str, cot_temperature: float, pal_temperature: float, backbone: str):
    '''
    This function is used to query OpenAI for answers in date understanding tasks. It contains three steps:
    1. Query CoT for solutions
    2. Query PAL for solutions
    3. Query model selection answers

    Note that we only query selection answers when CoT and PAL answers are different. Otherwise, we directly use CoT or PAL answers.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        cot_temperature: the temperature used in CoT.
        pal_temperature: the temperature used in PAL.
        backbone: ChatGPT or GPT-4

    Returns:
        to_dump_data: a dict containing the question, answer, the final answer and other information
    '''
    cot_ans = None
    pal_ans = None
    selection_ans = None
    final_ans = None
    cot_solution = query_cot(
        data, key, cot_temperature, backbone=backbone)
    if cot_solution is None:
        print('Time out')
        return None
    else:
        cot_ans = extract_date_cot(cot_solution[0])

    pal_solution = query_pal(
        data, key, pal_temperature, backbone=backbone)
    if pal_solution is None:
        print('Time out')
        return None
    else:
        pal_ans = execute_date_pal(pal_solution[0])

    if cot_ans is not None and pal_ans is not None:

        # ==== Only select when CoT and PAL are different ====
        if cot_ans != pal_ans:
            selection_ans = query_select(
                data, key, cot_solution=cot_solution, pal_solution=pal_solution, backbone=backbone)
            if selection_ans is None:
                print('Time out')
                return None
            else:
                selection_choice = extract_choice_turbo(selection_ans[0])
                if selection_choice == '(A)':
                    final_ans = cot_ans
                elif selection_choice == '(B)':
                    final_ans = pal_ans
        else:
            final_ans = cot_ans

    elif cot_ans is not None and pal_ans is None:
        final_ans = cot_ans
    elif cot_ans is None and pal_ans is not None:
        final_ans = pal_ans
    else:
        final_ans = None

    # === dump data ===
    to_dump_data = OrderedDict(
        {'index': data['index'], 'question': data['question'], 'answer': data['answer'],
         'final_ans': final_ans, 'cot_executed': cot_ans, 'pal_executed': pal_ans,
         'cot_generated': cot_solution, 'pal_generated': pal_solution, 'choice_solution': selection_ans}
    )

    return to_dump_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--backbone', type=str,
                        choices=['chatgpt', 'gpt4'], default='gpt4')
    parser.add_argument('--cot_temperature', type=float, default=0.)
    parser.add_argument('--pal_temperature', type=float, default=0.)
    parser.add_argument('--output_dir', type=str, default='../output/')
    parser.add_argument(
        '--key', type=str, default='sk-', required=True)

    args = parser.parse_args()

    start_index = args.start
    end_index = args.end
    cot_temperature = args.cot_temperature
    pal_temperature = args.pal_temperature
    backbone = args.backbone
    output_dir = args.output_dir
    key = args.key

    start_time_0 = time.time()
    print('Current time: ', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))

    dt_string = datetime.now().strftime("%m_%d_%H_%M")

    dataset = jsonlines_load('../dataset/date_understanding.jsonl')

    # === slice data based on start and end ===
    total_num = len(dataset)
    print('total data: ', total_num)
    if end_index == -1:
        end_index = total_num

    if end_index > total_num:
        end_index = total_num

    tasks = dataset[start_index:end_index]
    task_num = len(tasks)
    print('Current total tasks: ', task_num)

    unfinished_tasks = []

    output_path = os.path.join(output_dir, f'{backbone}/')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    save_path = os.path.join(output_path,
                             f'date_s{start_index}_e{end_index}_{dt_string}.jsonl')

    # === dump data ===
    progress_bar = tqdm(range(task_num))
    for i in range(task_num):
        task = tasks[i]
        start_time = time.time()
        while True:
            try:
                ans = query_date(
                    task, key=key, cot_temperature=cot_temperature,
                    pal_temperature=pal_temperature, backbone=backbone)
            except Exception as e:
                print(e)
                ans = None

            if ans is not None:
                with open(save_path, "a+") as fout:
                    fout.write(json.dumps(ans)+'\n')
                progress_bar.update(1)
                break
            else:
                sleep_time = random.uniform(3, 5)
                time.sleep(sleep_time)

            if time.time() - start_time > 120:
                print('Time out')
                print('Current Task: ', i)
                break

        sleep_time = random.uniform(3, 5)
        time.sleep(sleep_time)

    end_time_0 = time.time()
    print('Finish at time: ', time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime()))
    print(f'Time used: {end_time_0 - start_time_0} seconds')

    if len(unfinished_tasks) > 0:
        print('Unfinished tasks: ')
        for task in unfinished_tasks:
            print(task)

    print('Done')
