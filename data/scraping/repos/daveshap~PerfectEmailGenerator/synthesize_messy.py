import re
import os
import openai
from time import time,sleep
from uuid import uuid4


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


openai.api_key = open_file('openaiapikey.txt')

scenarios = open_file('baseline_intents.txt').splitlines()



def gpt3_completion(prompt, engine='text-davinci-002', temp=1.0, top_p=1.0, tokens=2000, freq_pen=0.0, pres_pen=0.0, stop=['asdfasdf', 'asdasdf']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


if __name__ == '__main__':
    for scenario in scenarios:
        prompt = open_file('prompt_messy.txt').replace('<<SCENARIO>>', scenario).replace('<<UUID>>', str(uuid4()))
        print('\n\n==========\n\n', prompt)
        completion = gpt3_completion(prompt)
        filename = scenario.replace(' ','').lower()[0:10] + str(time()) + '.txt'
        save_file('messyemails/%s' % filename, completion)
        print('\n\n', completion)