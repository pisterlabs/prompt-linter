import numpy as np
import pandas as pd
import numpy as np
import openai
from utils import get_response, create_prompt
from dotenv import load_dotenv
import os
from config import load_config

config = load_config()
prompts = config['prompts']
data_mode = config['data_mode']

load_dotenv()

api_key = os.getenv('API_KEY')
api_base = os.getenv('API_BASE')
api_type = os.getenv('API_TYPE')
api_version = os.getenv('API_VERSION')
deployment_name = os.getenv('DEPLOYMENT_NAME')

openai.api_key = api_key
openai.api_base = api_base
openai.api_type = api_type
openai.api_version = api_version
deployment_name = deployment_name

def parse_hypothesis(hypothesis):
    # hypothesis response is often in the form of "hypothesis: <hypothesis>"
    if 'Decision Tree:' in hypothesis:
        hypothesis = hypothesis.split('Decision Tree:')[1].strip()
    return hypothesis

def get_hypothesis(train_data, temperature=1, sample_size=16, num_hypotheses=1):
    system_content = prompts['SYSTEM_CONTENT_1']
    user_content_1 = prompts['USER_CONTENT_1']
    assistant_content_1 = prompts['ASSISTANT_CONTENT_1']
    user_content_2 = prompts['USER_CONTENT_2']
    ask_for_hypothesis = prompts['ASK_FOR_HYPOTHESIS']
    messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content_1}, {"role": "assistant", "content": assistant_content_1}, {"role": "user", "content": user_content_2}]
    prompt = create_prompt(sample_size, train_data=train_data, test_data=None, messages=messages, train_mode=True)
    prompt.append({"role": "user", "content": ask_for_hypothesis})
    # write prompt to text file
    with open(f'data/generated_prompts/hypothesis_prompt_{data_mode}.txt', 'w') as f:
        for el in prompt:
            f.write(el['role'] + ': ' + el['content'])
            f.write('\n\n')
    response = get_response(prompt, temperature, num_hypotheses)
    response = [parse_hypothesis(hypothesis) for hypothesis in response]
    return response

