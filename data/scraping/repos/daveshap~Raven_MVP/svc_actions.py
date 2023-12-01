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


def make_prompt(context):
    prompt = read_file('base_action_prompt.txt')
    return prompt.replace('<<CONTEXT>>', context)


def query_gpt3(context):
    prompt = make_prompt(context)
    response = openai.Completion.create(
        engine='davinci',
        #engine='curie',
        prompt=prompt,
        temperature=0.7,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0.7,
        presence_penalty=0.7,
        stop=['ACTION4:', 'CONTEXT:', 'INSTRUCTIONS:', '<<END>>'])
    return response['choices'][0]['text'].strip().splitlines()


def post_actions(actions, context):
    for action in actions:
        try:
            action = action.strip()
            if action == '':
                continue
            action = re.sub('ACTION\d+:', '', action)
            #print('ACTION:', action)
            payload = dict()
            payload['msg'] = action.strip()
            payload['irt'] = context['mid']
            payload['ctx'] = context['mid']
            payload['key'] = 'action.idea'
            payload['sid'] = 'action.generator'
            result = nexus_post(payload)
            #print(result)
        except Exception as oops:
            print('ERROR in ACTIONS/POST_ACTIONS:', oops)


def query_nexus():
    global last_msg
    try:
        stream = nexus_get(key='context', start=last_msg['time'])
        for context in stream:
            if context['time'] <= last_msg['time']:
                continue
            #print('CONTEXT:', context['msg'])
            if context['time'] > last_msg['time']:
                last_msg = context
            actions = query_gpt3(context['msg'])
            #print('ALL ACTIONS:', actions)
            post_actions(actions, context)
    except Exception as oops:
        print('ERROR in ACTIONS/QUERY_NEXUS:', oops)


if __name__ == '__main__':
    print('Starting Action Generator')
    while True:
        query_nexus()
        sleep(default_sleep)
