import os
import openai
import numpy as np
from time import time,sleep
from dotenv import load_dotenv
import pandas as pd
import json
import uuid
import math
import itertools

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    load_dotenv()  # take environment variables from .env.
except Exception as oops:
    print("Issue with load_dotenv:" + oops)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_convo(text, topic):
    with open('finetuning/%s_%s.txt' % (topic, time()), 'w', encoding='utf-8') as outfile:
        outfile.write(text)


openai.api_key = os.getenv("OPENAI_API_KEY")
prompt_version = os.getenv("PROMPT_VERSION")
base_prompt = open_file('syn_prompt2.txt')


def gpt3_completion(wdir, prompt, topic, engine='text-davinci-002', temp=1, top_p=1.0, tokens=3500, freq_pen=0.0, pres_pen=0.5, stop=['<<END>>']):
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
            print('lines = %s' % str(len(text.splitlines(True))))
            returned_lines = str(len(text.splitlines(True)))
            filename = '%s_gpt3.txt' % time()
            with open(wdir + '/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            response_info = '{"topic" : "%s",\nengine" : "%s",\ntemp" : "%s",\ntop_p" : "%s",\nfreq_pen" : "%s",\npres_pen" : "%s",\nreturned lines" : "%s" }' % (topic, engine, temp, top_p, freq_pen, pres_pen, returned_lines)
            data = response_info.split('\n')
            with open('gpt3_logs/%s' % 'data.log', 'a', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(0.25)

def compare_cosine(doclist):
    count_vect = CountVectorizer()
    for a, b in itertools.combinations(doclist, 2):
        corpus = [a,b]
        X_train_counts = count_vect.fit_transform(corpus)
        pd.DataFrame(X_train_counts.toarray(),columns=count_vect.get_feature_names(),index=['Document 1','Document 2'])
        vectorizer = TfidfVectorizer()
        trsfm=vectorizer.fit_transform(corpus)
        pd.DataFrame(trsfm.toarray(),columns=vectorizer.get_feature_names(),index=['Document 1','Document 2'])
        cosine_similarity(trsfm[0:1], trsfm)

if __name__ == '__main__':
    temp=1
    top_p=1.0
    tokens=3500
    freq_pen=0.0
    pres_pen=0.5

    topics = open_file('topics.txt').splitlines()
    vars_df = pd.read_csv('vars.csv')

    directory = 'gpt3_logs'
    loops = 0
    prompt_arr = {
        'Prompt':[], 
        'Topic': [], 
        'Response': []
    }

    #create a directory for this run in gpt3_logs
    filelist = filter(lambda x: (x.endswith('.run')), os.listdir(directory))
    print(filelist)
    myList = [i.split('.')[0] for i in filelist]
    print(myList)
    working_dir = str(int(max(myList, key=lambda x:float(x)))+1) + '.run'
    print('Creating %s\n' % working_dir)
    os.mkdir(directory + '\\' + working_dir)


    #set the new working directory based on the new working directory name
    directory = directory + '\\' + working_dir

    # for severity in vars_df["severity"]:
    #     if type(severity) != str:
    #         exit()
    #     print("Severity: %s\n" % severity)

    count=0

    store_df = pd.DataFrame()

    for temp in np.arange(0, 1, 0.2):
        print('Temp: %s\n' % temp)
        for top_p in np.arange(0, 1, 0.2):
            print('Top_p: %s\n' % top_p)
            for tokens in range(500, 3501, 3000):
                print('Tokens: %s\n' % tokens)
                count += 1

    print('Count %s' % count)

    short_topic_list = ['taking care of an old pet with medical issues', 'suicide']

    #for topic in vars_df["topic"]:
    for temp in np.arange(0, 1, 0.2):
        print('Temp: %s\n' % temp)
        for top_p in np.arange(0, 1, 0.2):
            print('Top_p: %s\n' % top_p)
            for tokens in range(500, 3501, 3000):
                print('Tokens: %s\n' % tokens)
                for topic in short_topic_list:
                    if type(topic) != str:
                        break
                    print("Topic: %s\n" % topic)
                    prompt = base_prompt.replace('<<TOPIC>>', topic)
                    utterance = "Human: Hi Daniel, what brings you here today?\nDaniel:"
                    prompt_arr['Prompt'].append(prompt)            
                    prompt_arr['Topic'].append(topic)            
                    prompt += utterance
                    utterance = utterance.replace("Human: ","")
                    utterance = utterance.replace("\nDaniel:","")
                    utterance = utterance.strip(" ")        
                    # prompt_arr['Utterance'].append(utterance)
                    # prompt_arr['temp'].append(temp)
                    # prompt_arr['top_p'].append(top_p)
                    # prompt_arr['tokens'].append(tokens)

                    #give it some salt
                    prompt = str(uuid.uuid4()) + '\n' + prompt

                    response = gpt3_completion(directory, prompt, topic)
                    prompt_arr['Response'].append(response)

                    new_row = [[prompt, topic, utterance, temp, top_p, tokens, response]]

                    store_df = store_df.append(pd.DataFrame(new_row, columns=['prompt', 'topic', 'utterance', 'temp', 'top_p', 'tokens', 'response']), ignore_index = True)

                    print('\n---------------------------------\n')
                    df = pd.DataFrame(data=prompt_arr)
                    print('df:')
                    print(df)
                    prompt_arr = {
                        'Prompt':[], 
                        'Topic': [], 
                        'Response': []
                    }
                    break
                    #df.to_json(directory + "\\%s__attributes.json" % (topic))
                break
            break
        break
    store_df.to_json(directory + "\\%s__attributes.json" % 'all')