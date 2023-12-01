import os
import openai
from time import time,sleep
from dotenv import load_dotenv
import pandas as pd
import json
import uuid
import argparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    load_dotenv()  # take environment variables from .env.
except Exception as oops:
    print("Issue with load_dotenv:" + oops)

parser = argparse.ArgumentParser(description="Process and view the scores in a folder. Defaults to the last run.")
parser.add_argument('-b', '--bad', action='store_true', help="Optional. Signify Bad actor therapist. Defaults to no.")
parser.add_argument('-n', '--number', type=int, help="Optional. Number of topics to generate. Defaults to 5.")

args = parser.parse_args()

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_convo(text, topic):
    with open('finetuning/%s_%s.txt' % (topic, time()), 'w', encoding='utf-8') as outfile:
        outfile.write(text)

openai.api_key = os.getenv("OPENAI_API_KEY")
prompt_version = os.getenv("PROMPT_VERSION")

if args.bad:    
    base_prompt = open_file('syn_prompt2 BAD.txt')
else:
    base_prompt = open_file('syn_prompt2.txt')

#gpt_model = 'gpt-3.5-turbo'
gpt_model = 'gpt-4'

def gpt4_completion(wdir, prompt, topic, engine=gpt_model, temp=1, top_p=1.0, tokens=3500, freq_pen=0.0, pres_pen=0.5, stop=['<<END>>']):
    max_retry = 5
    retry = 0
    while True:
        try:
            messages=[{"role": "user", "content": f'{prompt}'}]

            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['message']['content'].strip()
            print(f'{response["usage"]["prompt_tokens"]} prompt tokens used.')
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
                return "GPT error: %s" % oops
            print(f'Error communicating with OpenAI: {oops}\n')
            sleep(0.25)

def quality_check(run):
    filelist = filter(lambda x: (x.endswith('.json')), os.listdir(f'gpt3_logs/{run}.run'))
    for file in filelist:
        current_topic = open(file, 'r')
        print(f'{current_topic}\n')

if __name__ == '__main__':
    # default prompt attributes
    brothers = "no brothers"
    sisters = "one sister"
    severity = "mild"
    alone = "lives"
    topic = "depression"

    max_topics = args.number if args.number > 0 else 5

    #topics = open_file('topics.txt').splitlines()
    vars_df = pd.read_csv('vars.csv')

    first_utterance = open_file('utterances.txt').splitlines()
    utterance_loop = len(first_utterance)

    directory = 'gpt3_logs'
    raw_utterance = "\nUser: <<UTT>>\nDaniel:"
    prompt_arr = {
        'Prompt':[], 
        'Topic': [], 
        'Utterance': [],
        'Response': []
    }

    filelist = filter(lambda x: (x.endswith('.run')), os.listdir(directory))

    # Find the highest numbered directory
    highest_number = 0
    for file in filelist:
        try:
            number = int(file.rstrip('.run'))
            if number > highest_number:
                highest_number = number
        except ValueError:
            pass  # Ignore if the file name is not a number

    # Create a new directory with a number +1 higher than the highest
    new_dir_number = highest_number + 1
    new_directory = os.path.join(directory, f"{new_dir_number}.run")
    print('Creating %s\n' % new_directory)
    os.makedirs(new_directory)
    #set the new working directory based on the new working directory name
    directory = new_directory

    print(f'')

    num_topics = 0
    # for severity in vars_df["severity"]:
    #     if type(severity) != str:
    #         exit()

    for topic in vars_df["topic"]:
        if type(topic) != str:
            break
        print("Topic: %s\n" % topic)
        if num_topics > max_topics:
            break
        num_topics += 1

        for utterance in first_utterance:
            if type(utterance) != str:
                break
            print("Utterance: %s\n" % utterance)
            prompt = base_prompt.replace('<<TOPIC>>', topic)
            utterance = raw_utterance.replace('<<UTT>>', utterance)
            prompt_arr['Prompt'].append(prompt)            
            prompt_arr['Topic'].append(topic)            
            prompt_arr['Utterance'].append(utterance)
            prompt += utterance
            prompt = str(uuid.uuid4()) + '\n' + prompt

            response = gpt4_completion(directory, prompt, topic, engine="gpt-4")
            prompt_arr['Response'].append(response)

        print('\n---------------------------------\n')
        df = pd.DataFrame(data=prompt_arr)
        print('df:')
        print(df)
        prompt_arr = {
            'Prompt':[], 
            'Topic': [], 
            'Utterance': [],
            'Response': []
        }
        df.to_json(directory + "/%s_output.json" % (topic))
        #quality_check(new_dir_number)

            