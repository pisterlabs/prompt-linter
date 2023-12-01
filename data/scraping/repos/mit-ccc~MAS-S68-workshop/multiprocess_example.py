#!/usr/bin/env python3

"""
This demonstrates calling the OpenAI ChatGPT API with multiprocessing
to parallelize a batch of requests (and also with exponential-backoff retry logic so
that failed requests are retried)
"""

from tenacity import retry, stop_after_attempt, wait_random_exponential
from multiprocessing import Pool
import openai

_BATCH_SIZE = 10

_STATES = ['Alaska', 'Alabama', 'Arkansas', 'Arizona', 'California',
           'Colorado', 'Connecticut', 'District of Columbia', 'Delaware',
           'Florida', 'Georgia', 'Hawaii', 'Iowa', 'Idaho', 'Illinois',
           'Indiana', 'Kansas', 'Kentucky', 'Louisiana', 'Massachusetts',
           'Maryland', 'Maine', 'Michigan', 'Minnesota', 'Missouri',
           'Mississippi', 'Montana', 'North Carolina', 'North Dakota',
           'Nebraska', 'New Hampshire', 'New Jersey', 'New Mexico', 'Nevada',
           'New York', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Islant',
           'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah',
           'Virginia', 'Vermont', 'Washington', 'Wisconsin', 'West Virginia', 'Wyoming']

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def run_gpt_query(state):
    prompt = "The capital of %s is " % (state)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=20,
    )
    result = response["choices"][0]["message"]["content"]
    return result.replace(".", "")  # remove period

if __name__ == "__main__":
    with Pool(_BATCH_SIZE) as pool:
        pooled = pool.map(run_gpt_query, _STATES)
        for state, result in zip(_STATES, pooled):
            print("The capital of %s is %s!" % (state, result))
