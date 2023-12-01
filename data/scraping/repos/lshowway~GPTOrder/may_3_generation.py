"""
Write a pipeline
first step: read the data, including WinoGrand, SST2, GSM8K, TruthfulQA
second step: process the data, for classification and generation, input and label
third step: closed model, only api, chat, claude, and wenxinyiyan
fourth step: prompt, for different task, different models
fifth step: evaluation. metrics.

"""
from time import sleep

import anthropic
import numpy as np
import openai

from key import Chat_API, Claude_API
import pandas as pd

def access_api(model, temperature, max_tokens, top_k=2, prompt=None):
    if model == 'text-davinci-003':
        openai.api_key = Chat_API
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            n=1,
            stream=False,
            logprobs=top_k,
        )
    elif model == 'claude-v1.3':
        client = anthropic.Client(Claude_API)
        prompt = '{} {} {}'.format(anthropic.HUMAN_PROMPT, prompt, anthropic.AI_PROMPT)
        response = client.completion(
            prompt=prompt,
            model=model,
            temperature=temperature,  # 0 for choice, 1 for generation
            max_tokens_to_sample=max_tokens,
            top_k=top_k,
        )

    return response


def generate_XWinoGrade(data, model, lan, w_file, temperature, max_tokens, top_k=2, k=3):
    from may_3_prompt import winogrande_generation_prompt

    if model not in data.columns:
        data[model] = np.nan
        # data[model].fillna('', inplace=True)
        # data[model] = data[model].astype(str)

    for index, row in data.iterrows():
        if pd.isnull(data.loc[index, model]):
            label = row['answer']
            question = row['Question']
            option1 = row['option1']
            option2 = row['option2']
            if '_zh' in w_file:
                Q = "{}选项1: {}, 选项2: {}".format(question, option1, option2)
            else:
                Q = "{} option1: {}, option2: {}".format(question, option1, option2)
            prompt = winogrande_generation_prompt[lan].format(Q)
            k_answers = []
            for _ in range(k):
                while True:
                    try:
                        response = access_api(model, temperature, max_tokens=max_tokens, top_k=top_k, prompt=prompt)
                        break
                    except Exception as e:
                        print(e)
                        sleep(20)
                if model == 'text-davinci-003':
                    response = response['choices'][0]['text']
                    response = response.replace('\nA: ', '')
                elif model == 'claude-v1.3':
                    response = response['completion']
                print(response, label)
                print('================', Q)
                k_answers.append(response)
            data.loc[index, model] = str(k_answers)
            if (index + 1) % 5 == 0:
                data.to_csv(w_file, index=False, encoding='utf-8')
    data.to_csv(w_file, index=False, encoding='utf-8')


def generate_MGSM(data, lan, model, w_file, temperature, max_tokens, top_k=2, k=100):
    from may_3_prompt import MGSM_generation_prompt

    if model not in data.columns:
        data[model] = np.nan
        # data[model].fillna('', inplace=True)
        # data[model] = data[model].astype(str)

    for index, row in data.iterrows():
        if pd.isnull(data.loc[index, model]):
        # if data.loc[index, model] == '':
            question = row['Question']
            prompt = MGSM_generation_prompt[lan].format(question)
            k_answers = []
            for _ in range(k):
                while True:
                    try:
                        response = access_api(model, temperature, max_tokens=max_tokens, top_k=top_k, prompt=prompt)
                        break
                    except Exception as e:
                        print(e)
                        sleep(20)
                if model == 'text-davinci-003':
                    response = response['choices'][0]['text']
                    response = response.replace('\nA: ', '')
                elif model == 'claude-v1.3':
                    response = response['completion']
                print(response)
                print('================')
                k_answers.append(response)
            data.loc[index, model] = str(k_answers)
            if (index + 1) % 5 == 0:
                data.to_csv(w_file, index=False, encoding='utf-8')
    data.to_csv(w_file, index=False, encoding='utf-8')


def generate_TruthfulQA(data, model, w_file, temperature, max_tokens, top_k=2):
    from may_3_prompt import TruthfulQA_generation_prompt

    if model not in data.columns:
        data[model] = np.nan

    for index, row in data.iterrows():
        if pd.isnull(row[model]):
            question = row['Question']
            prompt = TruthfulQA_generation_prompt.format(question)
            response = access_api(model, temperature, max_tokens=max_tokens, top_k=top_k, prompt=prompt)
            if model == 'text-davinci-003':
                response = response['choices'][0]['text']
                response = response.replace('\nA: ', '')
            elif model == 'claude-v1.3':
                response = response['completion']
            print(response)
            print("-------------------------", model, w_file.split('/')[-1])
            data.loc[index, model] = response
            if (index + 1) % 20 == 0:
                data.to_csv(w_file, index=False, encoding='utf-8')
    data.to_csv(w_file, index=False, encoding='utf-8')


def generate_WiQueen(data, model, lan, w_file, temperature, max_tokens, top_k=2, k=3):
    from may_3_prompt import wiqueen_gen_prompt

    if model not in data.columns:
        data[model] = np.nan
        # data[model].fillna('', inplace=True)
        # data[model] = data[model].astype(str)

    for index, row in data.iterrows():
        if pd.isnull(data.loc[index, model]):
            relation = row['relation']
            e1 = row['e1']
            e2 = row['e2']
            e3 = row['e3']
            e4_candidates = row['e4_candidates']

            prompt = wiqueen_gen_prompt[lan].format(e1, e2, e3)
            k_answers = []
            for _ in range(k):
                while True:
                    try:
                        response = access_api(model, temperature, max_tokens=max_tokens, top_k=top_k, prompt=prompt)
                        break
                    except Exception as e:
                        print(e)
                        sleep(20)
                if model == 'text-davinci-003':
                    response = response['choices'][0]['text']
                    response = response.replace('\nA: ', '')
                elif model == 'claude-v1.3':
                    response = response['completion']
                print(response)
                print('================e1 {}, e2 {}, e3 {}, e4 {}'.format(e1, e2, e3, e4_candidates))
                k_answers.append(response)
            data.loc[index, model] = str(k_answers)
            if (index + 1) % 10 == 0:
                data.to_csv(w_file, index=False, encoding='utf-8')
    data.to_csv(w_file, index=False, encoding='utf-8')
