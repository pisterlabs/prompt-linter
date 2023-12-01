import csv
import random
import openai
from time import time,sleep
import re


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


openai.api_key = open_file('openaiapikey.txt')



def gpt3_completion(prompt, engine='text-davinci-002', temp=0.7, top_p=1.0, tokens=1000, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'TIM:']):
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
            #text = re.sub('\s+', ' ', text)
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
    reviews = list()
    with open('reviews.csv', 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            #print(row)
            if 'fire' in row[0].lower() and 'tablet' in row[0].lower() and 'charger' not in row[0].lower():
                reviews.append(row[1] + ' - ' + row[2])
    #print(len(reviews))
    #print(reviews[0])
    #print(reviews[-1])
    random.seed()
    summaries = list()
    for i in list(range(0,10)):
        subset = random.choices(reviews, k=25)
        textblock = '\n-'.join(subset)
        #print(len(subset))
        #print(len(textblock))
        prompt = open_file('prompt.txt').replace('<<REVIEWS>>', textblock)
        response = gpt3_completion(prompt)
        print('\n\n====================\n\n',response)
        summaries.append(response)
        #exit(0)
    textblock = '\n-'.join(summaries)
    prompt = open_file('prompt_improvements.txt').replace('<<REVIEWS>>', textblock)
    response = gpt3_completion(prompt)
    print('\n\n====================\n\n',response)