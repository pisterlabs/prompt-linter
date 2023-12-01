import os
import openai
from time import time,sleep


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


openai.api_key = open_file('openaiapikey.txt')


def save_file(content, filepath):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def gpt3_completion(prompt, engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=500, freq_pen=0.0, pres_pen=0.0, stop=['<<END>>']):
    max_retry = 5
    retry = 0
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
            filename = '%s_gpt3.txt' % time()
            with open('gpt3_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


if __name__ == '__main__':
    books = os.listdir('books/')
    for book in books:
        #if 'alice' in book:
        #    continue
        #if 'frankenstein' in book:
        #    continue
        #if 'greatgatsby' in book:
        #    continue
        #if 'pride' in book:
        #    continue    
        name = book.replace('.txt', '')
        chunks = [i for i in os.listdir('chunks/') if name in i]
        count = 0
        for chunk in chunks:
            count += 1
            #if count <= 11:
            #    continue
            prompt = open_file('prompt_summary.txt').replace('<<CHUNK>>', open_file('chunks/%s' % chunk))
            prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
            #print(prompt)
            response = gpt3_completion(prompt)
            print(name, chunk, response)
            save_file(response, 'summaries/%s' % chunk)
            if count >= 40:
                break
        