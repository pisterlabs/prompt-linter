# script inspired by those in https://github.com/ChloeL19/RLVF

from joblib import Memory
memory = Memory("cachegpt")
import openai
import sys
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_fixed
)  # for exponential backoff
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

openai.organization = "org-f2tK1brD8eM1W91o2X5WgNoy"
openai.api_key = config['OPENAI_KEY']

def default_model_if_none(model=None):
    if model is None:
        model = "gpt-3.5-turbo"
    return model

@retry(wait=wait_random_exponential(min=10, max=30), stop=stop_after_attempt(25))
def generate(messages, model=None): # "gpt-3.5-turbo", "gpt-4"
    model = default_model_if_none(model)
    print("calling GPT... model="+model)
    return openai.ChatCompletion.create(
        model=model,
        messages=messages)

@memory.cache
def ask(message, model=None):
    response = generate([{"role":"user", "content": message}], model=model)
    return response.choices[0].message.content

if __name__ == '__main__':
    fn_in = sys.argv[1]
    f_in = open(fn_in, 'r')
    message = f_in.read()
    f_in.close()
    output = ask(message)
    fn_out = sys.argv[2]
    f_out = open(fn_out, 'w')
    f_out.write(output)
    f_out.close()
    