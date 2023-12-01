import r2pipe
import openai

import os
import sys

# Usage:

# env:
    # export OPENAI_API_KEY='sk-32a1' (required) OpenAi api key
    # export OPENAI_API_MODEL='gpt-3.5-turbo' (optional) OpenAi model to use. Defaults to gpt-3.5-turbo

# commands:
    # !pipe python ./r2gpt.py

    # Can also use a custom prompt to which the assembly code will be appended to:
        # !pipe python ./r2gpt.py 'Why am I doing this?'

def main():
    if 'OPENAI_API_KEY' not in os.environ:
        print('Set an openai api key in enviroment variable OPENAI_KEY=')
        return
    
    model = 'gpt-3.5-turbo'
    if 'OPENAI_MODEL' in os.environ:
        model = os.environ['OPENAI_API_MODEL']
    

    r2 = r2pipe.open()

    r2.cmd('e scr.color=false')

    pdf = r2.cmd('pdf')

    r2.cmd('e scr.color=true')

    if 'ERROR:' in pdf or not pdf:
        print(pdf)
        return
    
    if len(sys.argv) > 1:
        messages = [{"role": "user", "content":  sys.argv[1] + '\n' + pdf}]
    else:
        messages = [{"role": "system", "content": "You will be provided with a piece of assembly code and your task is to explain in in a concise way."},{"role": "user", "content": pdf}]

    chat_completion = openai.ChatCompletion.create(model=model, messages=messages, stream=True)

    print('Please wait! This might take a while...')
    chat_completion = openai.ChatCompletion.create(model=model, messages=messages)

    print('\n', chat_completion['choices'][0]['message']['content'])



if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(Exception.__class__.__name__, exc, sep=' : ')