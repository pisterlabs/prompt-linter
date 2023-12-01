### Grab a pdf or an input and recursively summarize it using consice as the thingie
from argparse import ArgumentParser
import openai
import os
from time import time,sleep
import textwrap
import pdftotext
import re
import sys

## read 


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def open_pdf(filepath):
    with open(filepath, 'rb') as infile:
        pdf = pdftotext.PDF(infile)
    return '\n\n'.join(pdf)

def save_file(content, filepath):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

def gpt3_completion(prompt, engine='text-davinci-002', temp=0.6, top_p=1.0, tokens=2000, freq_pen=0.25, pres_pen=0.0, stop=['<<END>>']):
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
            text = re.sub('\s+', ' ', text)
            # filename = '%s_gpt3.txt' % time()
            # with open('gpt3_logs/%s' % filename, 'w') as outfile:
            #     outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)

openai.api_key = open_file('openaiapikey.txt')

if __name__ == '__main__':
    ## change the following line to open whatever file the user provides as an argument
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help = 'file to summarize', required = True)
    parser.add_argument('-o', '--output', help = 'output file', default = 'output.txt')
    args = parser.parse_args()
    filepath = args.input
    ## use the correct function to open the file whether it's a PDF or text
    if filepath.endswith('.pdf'):
        alltext = open_pdf(filepath)
    else:
        alltext = open_file(filepath)
    
    ## remove \r if present
    alltext = alltext.replace('\r', '')

    ## split the text into paragraphs
    paragraphs = alltext.split('\n\n')
    ## remove empty paragraphs
    paragraphs = [p for p in paragraphs if p.strip()]
    ## remove paragraphs that are too short
    paragraphs = [p for p in paragraphs if len(p) > 100]

    result = list()
    for para in paragraphs:
        prompt = open_file('prompt.txt').replace('<<SUMMARY>>', para)
        print(prompt)
        prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
        summary = gpt3_completion(prompt)
        print('\n\n\n', summary)
        result.append(summary)
    save_file('\n\n'.join(result), args.output)