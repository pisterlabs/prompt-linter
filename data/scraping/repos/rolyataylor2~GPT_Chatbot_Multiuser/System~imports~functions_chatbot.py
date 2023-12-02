#
#   Imports
#
import functions_helper as file
from time import time, sleep
import openai
import json
openai.api_key = file.open_file('api_key.txt')
gpt_scripts_directory = 'System/gpt-scripts'

#
#   GPT Handler
#
def execute(messages, model="gpt-3.5-turbo-16k-0613", temperature=0):
    if len(json.dumps(messages)) < 4000:
        model = "gpt-3.5-turbo-0613"
    
    max_retry = 7
    retry = 0
    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
            text = response['choices'][0]['message']['content']
            return text
        except Exception as oops:
            print(f'\n\nError communicating with OpenAI: "{oops}"')
            if 'maximum context length' in str(oops):
                a = messages.pop(1)
                print('\n\n DEBUG: Trimming oldest message')
                continue
            retry += 1
            if retry >= max_retry:
                print(f"\n\nExiting due to excessive errors in API: {oops}")
                exit(1)
            print(f'\n\nRetrying in {2 ** (retry - 1) * 5} seconds...')
            sleep(2 ** (retry - 1) * 5)
def getScript(name):
    content = file.open_file(gpt_scripts_directory + '/' + name + '.txt')
    return content


