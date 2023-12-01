import openai
import os
import time

from sympy import isprime
from random import randint

openai.api_key = os.environ.get('OPENAI_API_KEY')

MODELS = ['gpt-3.5-turbo-0301', 'gpt-3.5-turbo-0613', 'gpt-4-0314', 'gpt-4-0613']

PROMPT = 'Is {} a prime number? Think step by step and then answer "[Yes]" or "[No]"'

def evaluate(model, number):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": PROMPT.format(number)}
            ],
            temperature=0.1,
            max_tokens=1000,
        )
    except Exception as e:
        print(f"{e} waiting 10 seconds")
        time.sleep(10)
        return evaluate(model, number)
    return response['choices'][0]['message']['content']

def extract_answer(response):
    if '[yes]' in response.lower():
        return True
    elif '[no]' in response.lower():
        return False
    else:
        return None

def score_response(response, reference):
    answer = extract_answer(response)
    if answer is None:
        return 0
    else:
        return int(answer == reference)
    
def run_experiment(model):
    lower, upper = 10000, 100000
    number = randint(lower, upper)
    response = evaluate(model, number)
    is_prime = isprime(number)
    score = score_response(response, is_prime)
    return [model, number, is_prime, score, response]

if not os.path.exists('results.csv'):
    with open('results.csv', 'w') as f:
        f.write('model,number,is_prime,score\n')

N = 1000
runs = []

with open('results.csv', 'a') as f, open('responses.csv', 'a') as g:
    for i in range(N):
        print(f"iteration: {i}", end='\r')
        for model in MODELS:
            run = run_experiment(model)
            f.write(','.join([str(x) for x in run[:-1]]) + '\n')
            f.flush()
            g.write(run[-1] + '\n\n')
            g.write('-'*50 + '\n\n')
            g.flush()
            runs.append(run)