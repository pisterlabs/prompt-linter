import openai
import time
import json
import argparse
import pandas as pd
import os
from more_itertools import chunked, flatten
from isl_utils import *
import os
from tqdm import tqdm

openai.api_key = os.environ.get('OPENAI_KEY')

GPT_MODEL = "gpt-4"

prompt_begin = """
I'm constructing a list of alternate phrases for an Indian Sign Language Dictionary. This dictionary contains a lot
of words/phrases that are specific to the Indian Subcontinent. It also has words that describe various aspects of 
life, such as vocation, family, food, culture, health, legalese etc. Some words are really specific (to banking or zoology) 
and may not be used very often. Others may be in "Hinglish", an amalgamation of Hindi and English. I want you to offer 
alternate phrases that may be used in place of these words. For proper nouns, make sure to describe what they represent. 
Here is an example input/output.

Input: 

{
    "38d2a2": ["loud", "high volume"],
    "das124": ["the"],
    "s103bf": ["Virat Kohli"],
    "5320fa": ["hard disk", "hard drive"],
    "132111": ["model"]
}


Output: 

{
    "38d2a2": ["raucous"],
    "das124": [], 
    "s103bf": ["Indian Cricketer", "Virat", "Kohli", "Former Indian cricket captain"],
    "5320fa": ["persistent storage for computer"],
    "132111": []
}
    
Explanation:

The keys in the JSON are a hash. "raucous" is a synonym for "loud". "the" is unique, doesn't really have synonyms. 
"Virat Kohli" is a proper noun (refers to popular Indian Sportsperson). "hard disk" and "hard drive" refer
to "persistent storage for computer". Finally, if the context is not clear, such as with "model", just give an
empty list. Give upto 5 options for each.

Now please process the following input:

""" 
prompt_end = """

Please give me just the JSON and nothing else.
"""

def prepare_prompt (dictionary) :
    json_string = json.dumps(dictionary, indent=2)
    return prompt_begin + json_string + prompt_end

@skip_if_processed()
def process_chunk(data) : 
    i, chunk = data
    while True: 
        try :
            prompt_dict = {}
            reverse_map = {}
            for k, v in chunk.items() :
                prompt_dict[k[:6]] = v
                reverse_map[k[:6]] = k
            prompt = prepare_prompt(prompt_dict)
            response = openai.ChatCompletion.create( 
                model=GPT_MODEL,
                messages=[
                    {'role': 'system', 'content': 'You help build a dictionary/thesaurus for Indian Sign Language'},
                    {'role': 'user', 'content': prompt},
                ],
            )
            resp = response['choices'][0]['message']['content']
            resp_as_dict = json.loads(resp)
            if not (set(prompt_dict.keys()) == set(resp_as_dict.keys())) :
                continue
            final_dict = {}
            for k, v in resp_as_dict.items() :
                final_dict[reverse_map[k]] = v
            with open(str(i).zfill(6) + '_' + args.out_json, 'w+') as fp :
                json.dump(final_dict, fp, indent=2)
            time.sleep(5)
            break
        except Exception as e :
            print(e)
            continue

def main (args) : 
    hash_to_text_mapping = pd.read_csv(HASH_TO_TEXT_MAPPING)
    grouped = hash_to_text_mapping.groupby(['hash', 'original_text'])
    pairs = [(group['hash'].iloc[0], group['text'].tolist()) for name, group in grouped]
    chunked_pairs = chunked(pairs, 10)
    dict_chunks = list(map(dict, chunked_pairs))
    datas = list(enumerate(dict_chunks))
    list(pmap(process_chunk, datas))

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser(description="Expand the set of phrases by adding synonyms and the like")
    parser.add_argument('--out_json', required=True, type=str, help="Path to the video json directory.")
    args = parser.parse_args()
    main(args)

