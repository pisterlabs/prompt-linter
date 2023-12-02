# -*- coding: utf-8 -*-
import os
import math
import json
import time

import dotenv
import openai
import requests
import jsonlines as jsl

from tqdm import tqdm

from common import remove_punctuation, clean, AVERAGE_SENTENCE_LENGTH

uk = {
    'should': {
        'sing': "має",
        'pl': "мають"
    }
}

if __name__ == '__main__':
    dotenv.load_dotenv()

    openai.api_key = os.getenv('SECRET_KEY')

    generated = []

    default_prompt = "Напиши {0} різних завдань до даного тексту для перевірки розуміння прочитаного. У кожному завданні має бути одне запитання, пронумероване арабськими цифрами (1, 2, 3 ...). З цих {0} завдань {1} містити два варіанти відповіді, {2} містити три варіанти відповіді, {3} містити чотири варіанти відповіді. Варіанти відповіді повинні мати вигляд переліку, позначеного буквами (а, б, в, г). З усіх варіантів другий варіант (б) завжди має бути правильною відповіддю. У кожному завданні правильною має бути лише одна відповідь."
    
    records = jsl.Reader(open('edited_gpt3.jsonl'))

    ts = int(time.time())

    records = list(records)

    NUM_CAT = 3

    with jsl.open('generated_{}.jsonl'.format(ts), 'w') as writer:
        records = list(enumerate(records))
        for record_idx, record in tqdm(records):
            if record_idx <= 37:
                continue
            text = record['text']
            words = remove_punctuation(clean(text)).split()
            num_q = max(NUM_CAT, int(math.ceil(len(words) / AVERAGE_SENTENCE_LENGTH)))
            num_q_base = num_q // NUM_CAT
            num_q_remainder = num_q % NUM_CAT
            # Order: 2, 3, 4 alternatives
            num_alt = [num_q_base] * NUM_CAT
            for i in range(num_q_remainder):
                num_alt[NUM_CAT - i - 1] += 1

            for i in range(NUM_CAT):
                if num_alt[i] % 10 == 1:
                    num_alt[i] = "{} {}".format(num_alt[i], uk['should']['sing'])
                else:
                    num_alt[i] = "{} {}".format(num_alt[i], uk['should']['pl'])

            prompt = default_prompt.format(num_q, *num_alt)
            
            max_tokens = 2000
            while True:
                try:
                    gen_params = {
                        'prompt': '{}\n\n{}\n\n'.format(
                            prompt, text
                        ),
                        'temperature': 0.7,
                        'max_tokens': max_tokens
                    }
                    completion = openai.Completion.create(engine='text-davinci-003', **gen_params)
                except openai.error.RateLimitError:
                    time.sleep(60)
                    continue
                except openai.error.InvalidRequestError:
                    max_tokens -= 100
                    print("Downgraded max_tokens to {}".format(max_tokens))
                    continue

                writer.write({
                    'text': text,
                    'requested_q': num_q,
                    'num_alt': num_alt,
                    'prompt': prompt,
                    'params': gen_params,
                    'res': completion
                })

                break
