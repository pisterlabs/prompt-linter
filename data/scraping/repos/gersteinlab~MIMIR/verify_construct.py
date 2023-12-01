# -*- coding: UTF-8 -*-
"""
@Author: ryannnwang
@CreateDate: 2023-07-05
@File: data_verification.py
@Project: Yale summer
"""
import os
import sys
import ast
from pathlib import Path
import logging
from typing import Union, List, Dict, Mapping
import time

ROOT = Path(__file__).absolute().parents[1]
sys.path.append(str(ROOT))

import json
from conf.instruction_config import INSTRUCTION_VERIFY
import openai, json, random
from conf.config import *

openai.api_type = "azure"
openai.api_base = configure["open_ai_api_base"]
openai.api_version = "2023-05-15"
# get this API key from the resource (its not inside the OpenAI deployment portal)
openai.api_key = configure["open_ai_api_key"][0]
key_bundles =  configure["key_bundles"]

def load_json_list(fp: Union[Path, str], encoding='utf-8') -> List[dict]:
    """load a txt file where each row as a dict.

    :param fp: the file path of the json-lines
    :param encoding: the encoding manner

    :return: a list of dicts
    """
    with open(fp, encoding=encoding) as f:
        json_lines = list(map(lambda x: json.loads(x), f.readlines()))
    return json_lines


def verify(q4human: str, a4ai: str) -> Dict[str, str]:
    # res_list = []
    input_txt = 'Question: ' + q4human + '\n' + 'Answer: ' + a4ai
    narration = make_narration(instruction=INSTRUCTION_VERIFY, 
                                input_text=input_txt, 
                                temperature=0.1,
                                model_type='chatgpt')
    return narration


def make_narration(instruction=None, *args, **kwargs):
    temp = kwargs.get('temperature', 0.1)
    input_text = kwargs['input_text']

    input_text = instruction + '\n' + input_text
    input_msg = [{"role": "user", "content": input_text}]

    try:
        response = openai.ChatCompletion.create(engine="gpt-4", messages=input_msg, temperature=temp)
        result = response.choices[0].message["content"]
        return result
        #print(result)
    except openai.error.Timeout:
        print("Timeout")
        key_bundle = random.choice(key_bundles)
        openai.api_key, openai.api_base = key_bundle
        response = openai.ChatCompletion.create(engine="gpt-35-turbo", messages=input_msg, temperature=temp)
        result = response.choices[0].message["content"]
        #print(result)
    except openai.error.InvalidRequestError:
        print("InvalidRequestError")
        #print(result)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        #print(result)


def iter_jsonl(fp: Union[Path, str], encoding='utf-8'):
    """load a txt file where each row as a dict.

    :param fp: the file path of the json-lines
    :param encoding: the encoding manner

    :return: a list of dicts
    """
    with open(fp, encoding=encoding) as f:
        line = f.readline()
        while line.strip():
            dct = json.loads(line)
            yield dct
            line = f.readline()


def dump_json_list(lst_of_dct: List[Mapping], fp, encoding='utf-8'):
    with open(fp, 'w', encoding=encoding) as f:
        f.writelines(
            [json.dumps(dct, ensure_ascii=False) + '\n' for dct in lst_of_dct]
        )


def __test__():
    # user_msg = "Tell me something about california University LA."
    user_msg = """
    Question: I'm interested in learning more about biomedical research. Can you tell me about any recent breakthroughs in the field?
    Answer: Yes, there have been many recent breakthroughs in computer research. Here are a few examples:
    1. GPT-3: In 2020, OpenAI released GPT-3, a language model with 175 billion parameters that can generate \
    human-like text, answer questions, and perform tasks like translation and summarization.
    2. AlphaFold: In 2020, DeepMind's AlphaFold was able to predict the 3D structure of proteins with remarkable \
    accuracy, which could revolutionize drug discovery and other fields.
    3. Quantum Computing: In 2021, Google announced that its Sycamore quantum processor had achieved "quantum supremacy," \
    performing a calculation that would take the world's fastest supercomputer thousands of years to complete.
    """
    result = make_narration(instruction=INSTRUCTION_VERIFY, input_text=user_msg, temperature=0.1)
    print("result: ", result)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s-%(lineno)d-%(funcName)s(): '
               '%(levelname)s\n %(message)s')
    t = time.time()

    __test__()

    print('Done running file: {}\nTime: {}'.format(
        os.path.abspath(__file__), time.time() - t,
    ))
