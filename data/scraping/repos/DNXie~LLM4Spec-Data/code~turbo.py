import os
import openai
from util import *
import json 
import argparse
import time

TOKEN_LIMIT=4096


def call_gpt(json_fpath, index, save_path, key, max_tokens=100):
    key = os.getenv("OPENAI_API_KEY")  # overwrite key


    openai.api_key = key
    data = read_json(json_fpath)
    prompt = data['data'][int(index)]['prompt'].strip()

    save_data = data['data'][int(index)]
    while True:
        try:
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=['\n\n###\n\n']
            )
            end_time = time.time()
            break
        except openai.error.InvalidRequestError:
            prompt, cnt = cut_prompt_from_top(prompt, TOKEN_LIMIT)
            save_data['new_prompt'] = prompt
        except:
            exit(1)
    


    save_data['response'] = json.loads(json.dumps(response)) 
    save_data['time'] = end_time - start_time

    answer = response['choices'][0]['message']['content']

    
    save_data['answer'] = answer

    dump_json(save_path, save_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt_path')
    parser.add_argument('index')
    parser.add_argument('save_path')
    parser.add_argument('key')
    parser.add_argument('--max_tokens', default=100,
                        help='token limit for answer')
    args = parser.parse_args()
        
    call_gpt(args.prompt_path, args.index, args.save_path, args.key, int(args.max_tokens))

