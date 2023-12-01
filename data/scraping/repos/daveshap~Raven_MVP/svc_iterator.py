import re
import os
import openai
import requests
import json
from time import sleep
from functions import *
import urllib3


default_sleep = 5
urllib3.disable_warnings()
open_ai_api_key = read_file('openaiapikey.txt')
openai.api_key = open_ai_api_key
last_msg = {'time':0.0}
iterated_actions = list()
context_action_limit = 15


def make_prompt(context, action, cof1, cof2, cof3):
    prompt = read_file('base_iterator_prompt.txt')
    prompt = prompt.replace('<<CONTEXT>>', context)
    prompt = prompt.replace('<<ACTION>>', action)
    prompt = prompt.replace('<<COF1>>', cof1)
    prompt = prompt.replace('<<COF2>>', cof2)
    prompt = prompt.replace('<<COF3>>', cof3)
    return prompt


def query_gpt3(context, action, cof1, cof2, cof3):
    prompt = make_prompt(context, action, cof1, cof2, cof3)
    response = openai.Completion.create(
        engine='davinci',
        prompt=prompt,
        temperature=0.5,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0.3,
        presence_penalty=0.3,
        stop=['<<END>>', 'CONTEXT:', 'ACTION:', 'NEW4:'])
    return response['choices'][0]['text'].strip().splitlines()


def post_actions(actions, context):
    for action in actions:
        try:
            action = action.strip()
            if action == '':
                continue
            action = re.sub('NEW\d+:', '', action)
            payload = dict()
            payload['msg'] = action.strip()
            payload['irt'] = context['mid']
            payload['ctx'] = context['mid']
            payload['key'] = 'action.idea'
            payload['sid'] = 'action.iterator'
            result = nexus_post(payload)
        except Exception as oops:
            print('ERROR in ITERATOR/POST_ACTIONS:', oops)


def get_cof_evalutions(action):
    try:
        irt = nexus_get(irt=action['mid'])
        cof1 = [i for i in irt if i['sid'] == 'cof1.evaluation'][0]
        cof2 = [i for i in irt if i['sid'] == 'cof2.evaluation'][0]
        cof3 = [i for i in irt if i['sid'] == 'cof3.evaluation'][0]
        return cof1, cof2, cof3
    except Exception as oops:
        return None, None, None


def query_nexus():
    global last_msg
    try:
        actions = nexus_get(key='action.idea')
        actions = [i for i in actions if i['mid'] not in iterated_actions]
        for action in actions:
            total_actions = nexus_get(ctx=action['ctx'], key='action.idea')
            if len(total_actions) > context_action_limit:
                continue  # if there are a lot of actions for this context already, skip ahead
            cof1, cof2, cof3 = get_cof_evalutions(action)  # TODO prioritize higher scoring actions
            if cof1 and cof2 and cof3:  # TODO support multiple COF services
                context = nexus_get(mid=action['ctx'])
                #print('CONTEXT:', context)
                #print('COF1:', cof1)
                #print('COF2:', cof2)
                #print('COF3:', cof3)
                iterations = query_gpt3(context[0]['msg'], action['msg'], cof1['msg'], cof2['msg'], cof3['msg'])
                post_actions(iterations, context[0])
                iterated_actions.append(action['mid']) 
    except Exception as oops:
        print('ERROR in ITERATOR/QUERY_NEXUS:', oops)


if __name__ == '__main__':
    print('Starting Iterator')
    while True:
        query_nexus()
        sleep(default_sleep)
