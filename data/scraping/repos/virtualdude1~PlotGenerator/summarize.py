import re
import os
import textwrap

import openai
import string
from time import time, sleep
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


def create_id():
    import random
    x = ''.join(random.choice(string.ascii_letters + string.digits)
                for i in range(12))
    return str(x).strip('.').replace('.', '_').replace('-', '_')


def open_file(filename):
    with open(filename, 'r') as f:
        return f.read()


def save_file(filename, text):
    with open(filename, 'w') as f:
        f.write(text)
def run_prompt(filename, story):
    prompt = open_file(filename).replace('<<SUMMARY>>', story)
    return sum_completion(prompt)

def sum_completion(prompt, engine='text-davinci-002', temp=0.79, best_of=1, top_p=1.0, tokens=1500, freq_pen=0.0,
                    pres_pen=0.0, stop=['asdfasdf', 'asdasdf']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()  # force it to fix any unicode errors
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                best_of=best_of,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3' % time()
            filename.strip('.').replace('.', '_')
            filename: str = filename + '.txt'
            save_file('summary_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)

if __name__ == '__main__':
    transcript = open_file('sum/episode002.txt').strip('\n\n')
    save_file('current_id.txt', create_id())
    transcript_id = create_id()

    chunk_size = 1500
    chunks = [transcript[i:i + chunk_size] for i in range(0, len(transcript), chunk_size)]
    # number of chunks print (f'Chunks: {len(chunks)}') outputs: Chunks: 22
    results = list()
    for chunk in chunks:
        transcript_id = create_id()
        save_file('current_id.txt', transcript_id)
        prompt = open_file('p_sum.txt').replace('<<SUMMARY>>', chunk).replace('<<ID>>', transcript_id)
        outputs = sum_completion(prompt)
        print(f'Preview: {outputs[:100]}')
        results.append(outputs)
        print(f'Completed {len(results)} of {len(chunks)}')
        save_file('current_id.txt', transcript_id)
        filename = 'sum/%s.txt' % transcript_id
        save_file(filename, outputs)
        print('Saved to %s' % filename)
    output = '\n'.join(results)
    save_file('sum/%s.txt' % transcript_id, output)
    print(f' length: {len(output)})')
    print(f'Final output: {output}')
    if len(output) > 1000:
        open_file('current_id.txt')
        open_file('sum/%s.txt' % transcript_id)
        sum_completion(open_file('p_sum.txt').replace('<<SUMMARY>>', output).replace('<<ID>>', transcript_id), tokens =2500)
        filename = 'sum/%s.txt' % transcript_id
        save_file(filename, output)
        print('Saved to %s' % filename)
    save_file('sum/'+transcript_id+'.txt', output)



