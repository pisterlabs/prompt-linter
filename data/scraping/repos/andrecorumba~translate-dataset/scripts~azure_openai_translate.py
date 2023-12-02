import os
import openai
from dotenv import load_dotenv, find_dotenv

import tqdm
import json

import datasets

import time


def azure_openai_translate():
    response_folder = 'response'
    log_folder = 'log'
    
    # Load dataset
    ds = my_load_dataset()

    # Itarate over the dataset
    count_error = 0
    idx_start = 0

    for idx, row in enumerate(tqdm.tqdm(ds)):
        if idx >= idx_start:
            #row['text_eng'] = '\n'.join( translate(s) for s in row['text'].split('\n'))
            row['text_eng'] = ''
            for s in row['text'].split('\n'):
                try: 
                    row['text_eng'] += translate(s) + '\n'

                except Exception as e:
                    row['text_eng'] = row['text']
                    # errors log
                    with open(os.path.join(log_folder,'error.log'), 'a') as f:
                        f.write(f"{idx:>011} - {e}\n{s}\n\n")
                    count_error += 1
                    break
            
            # create a new json file
            with open(os.path.join(response_folder,f'{idx:>011}.json'), 'w') as f:
                json.dump(row, f)

            # Verify if the limit of 3500 requests per minute was reached
            if idx % 3500 == 0:
                time.sleep(65)
            # if idx >= 10:
            #     break

    # Count errors
    with open(os.path.join(log_folder,'error_count.log'), 'w') as f:
        f.write(f"Total of errors: {count_error}\n")
    
def my_load_dataset():
    '''Load dataset'''
    data_folder = 'data'
    ds = datasets.load_dataset('json',
                           data_files=os.path.join(data_folder, 'treinamento_v01_full.json'),
                           split='train',
                           streaming=False)
    return ds

def translate(text):
    '''Translate text to English'''

    # Use Azure environment
    _ = load_dotenv(find_dotenv())
    openai.api_key = os.environ['AZURE_OPENAI_KEY']
    openai.api_base = os.environ['AZURE_OPENAI_ENDPOINT']
    openai.api_type = 'azure'
    openai.api_version = '2023-05-15'

    # Translate with Azure OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=1,
        engine='Teste',
        messages=[{"role": "user", 
                   "content": f"Translate to English the text bellow. If there is no text below for you to translate, write a dot. text: {text}"}]
        )

    return response.choices[0].message.content

def union_responses():
    '''Union all json files in one'''
    response_folder = 'response'
    ds = datasets.load_dataset('json', 
                               data_files=os.path.join(response_folder,'*.json'), 
                               split='train', 
                               streaming=False)
    ds.to_json(os.path.join(response_folder,'traducao_teste.json'), orient='records')

if __name__ == '__main__':
    print('Starting...')
    azure_openai_translate()
    print('Done!')

    print('Union responses...')
    union_responses()
    print('Done!')