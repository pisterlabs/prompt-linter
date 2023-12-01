import numpy as np
import pandas as pd
import numpy as np
import openai
from utils import get_response, create_prompt, parse_response
from dotenv import load_dotenv
import os
from config import load_config
from data_loader import split_data
import multiprocessing

config = load_config()
prompts = config['prompts']
# probably should put this in the config file
TIMEOUT = 10


def get_response_with_timeout(prompt, temperature):
    return get_response(prompt, temperature)

def get_few_shot_prompt(icl_data, test_data):
    system_content = prompts['SYSTEM_CONTENT_2']
    assistant_content_1 = prompts['ASSISTANT_CONTENT_1']
    user_content_2 = prompts['USER_CONTENT_2']
    messages = [{"role": "system", "content": system_content}, {"role": "assistant", "content": assistant_content_1}, {"role": "user", "content": user_content_2}]
    prompt = create_prompt(icl_data=icl_data, test_data=test_data, messages=messages, train_mode=True, test_mode=True)
    return prompt

def few_shot_one_example(icl_data, test_validation_data):
    assert 'Label' in test_validation_data.keys(), 'Few Shot One Example takes one example at a time'
    test_icl_data_keys = list(icl_data.keys())
    np.random.shuffle(test_icl_data_keys)
    test_icl_data_keys = test_icl_data_keys
    icl_data = {key: icl_data[key] for key in test_icl_data_keys}
    test_label = test_validation_data['Label']
    prompt = get_few_shot_prompt(icl_data, test_validation_data)
    print(prompt[-1]['content'])
    response_text = get_response(prompt, temperature=0, timeout=TIMEOUT)[0]
    response = parse_response(response_text)
    if response == test_label:
        correct = 1
    else:
        if response == -1:
            correct = -1
        else:
            correct = 0
    return {'response': response, 'correct': correct, 'label': test_label, 'text': response_text}

