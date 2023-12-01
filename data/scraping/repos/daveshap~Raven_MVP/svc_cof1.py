import re
import os
import openai
import requests
import json
from time import sleep
from functions import *
import urllib3


default_sleep = 1
urllib3.disable_warnings()
open_ai_api_key = read_file('openaiapikey.txt')
openai.api_key = open_ai_api_key
last_msg = {'time':0.0}


def make_prompt(context, action):
    prompt = read_file('base_cof1_prompt.txt')
    return prompt.replace('<<CONTEXT>>', context).replace('<<ACTION>>', action)


def query_gpt3(context, action):
    prompt = make_prompt(context, action)
    response = openai.Completion.create(
        engine='davinci',
        #engine='curie',
        prompt=prompt,
        temperature=0.7,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0.7,
        presence_penalty=0.7,
        stop=['<<END>>', 'CONTEXT:', 'ACTION:', 'INSTRUCTIONS:'])
    return response['choices'][0]['text']


def query_nexus():
    global last_msg
    try:
        stream = nexus_get(key='action.idea', start=last_msg['time'])
        for action in stream:
            if action['time'] <= last_msg['time']:
                continue
            #print('ACTION:', action['msg'])
            if action['time'] > last_msg['time']:
                last_msg = action
            context = nexus_get(mid=action['irt'])
            evaluation = query_gpt3(context[0]['msg'], action['msg'])
            evaluation = re.sub('\s+', ' ', evaluation).strip()
            #print('EVAL:', evaluation)
            if 'positive EXPLAIN:' in evaluation:
                evaluation = evaluation.replace('positive EXPLAIN:', '').strip()
                payload = {'msg': evaluation, 'irt': action['mid'], 'key': 'cof1.positive', 'sid': 'cof1.evaluation', 'ctx': action['ctx']}
            else:
                evaluation = evaluation.replace('negative EXPLAIN:', '').strip()
                payload = {'msg': evaluation, 'irt': action['mid'], 'key': 'cof1.negative', 'sid': 'cof1.evaluation', 'ctx': action['ctx']}
            result = nexus_post(payload)
            #print(result)
    except Exception as oops:
        print('ERROR in COF1/QUERY_NEXUS:', oops)


if __name__ == '__main__':
    print('Starting COF1')
    while True:
        query_nexus()
        # TODO look for metacognitive messages (aka HALT and RESUME)
        sleep(default_sleep)
