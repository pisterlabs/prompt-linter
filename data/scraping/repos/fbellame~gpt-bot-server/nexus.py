import os
import tools as tools
from openai_wrapper import gpt3_embedding, gpt3_completion
from time import time
from uuid import uuid4
import i18n

#
# Nexus methods for conversation
# 
#

PROMPT_RESPONSE = i18n.translation('PROMPT_RESPONSE', i18n.LANGUAGE)
PROMPT_RESPONSE_WITH_NOTE = i18n.translation('PROMPT_RESPONSE_WITH_NOTE', i18n.LANGUAGE)
USER            = i18n.translation('USER', i18n.LANGUAGE)

def load_convo():
    files = os.listdir('nexus')
    files = [i for i in files if '.json' in i]  # filter out any non-JSON files
    result = list()
    for file in files:
        data = tools.load_json('nexus/%s' % file)
        result.append(data)
    #ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    return result

def save_conversation(speaker, msg, conversations):
    message = '%s: %s' % (speaker, msg)

    info = {'speaker': speaker, 'message': message}
    timestamp = time()
    filename = 'log_%s_%s.json' % (timestamp, speaker)
    tools.save_json('nexus/%s' % filename, info)
    conversations.append(info)

    return conversations

def get_last_messages(conversation, limit):
    try:
        short = conversation[-limit:]
    except:
        short = conversation
    output = ''

    # walk through message list to make it a string
    for i in short:
        output += '%s\n\n' % i['message']
    output = output.strip()
    return output

def fetch_memories(vector, logs, count):
    scores = list()
    for i in logs:
        if vector == i['vector']:
            # skip this one because it is the same message
            continue
        score = tools.similarity(i['vector'], vector)
        i['score'] = score
        scores.append(i)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    # TODO - pick more memories temporally nearby the top most relevant memories
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered

def ask_theo(user_msg, rep_prompt, max_token, temperature, contexte):
    prompt = user_msg
    if rep_prompt == 'thera_prompt':
        prompt = tools.open_file('promts/%s' % PROMPT_RESPONSE).replace('<<%s>>' % USER, user_msg)
        #### generate response, vectorize, save, etc
    elif rep_prompt == 'thera_prompt_with_context':
        prompt = tools.open_file('promts/%s' % PROMPT_RESPONSE_WITH_NOTE).replace('<<CONTEXTE>>', contexte).replace('<<%s>>' % USER, user_msg)

    output = gpt3_completion(prompt, temp=temperature, top_p=1.0, tokens=max_token)

    return output