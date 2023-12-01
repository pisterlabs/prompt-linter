#!/usr/bin/env python
import openai
import argparse
import gpt
from prompt_toolkit import PromptSession, print_formatted_text, HTML
from prompt_toolkit.key_binding import KeyBindings

def parse_args():
    parser = argparse.ArgumentParser(description='Run GPT-3 from the terminal.')
    parser.add_argument('--model', type=str, help='The GPT-3 model that will be called.', default='text-davinci-003')
    parser.add_argument('--max-tokens', type=str, help='The max tokens to return.', default='2000')
    parser.add_argument('--temperature', type=str, help='The temperature to use.', default='0.2')
    parser.add_argument('--api_key', type=str, help='Your OpenAI API Key. Alternatively, set a OPENAI_API_KEY environment variable.', default='')

    
    return parser.parse_args()



def welcome_message():
    print_formatted_text(HTML('''


            #--------------------------#
            #----                  ----#
            #------ <b>Simple GPT-3</b> ------#
            #----                  ----#
            #--------------------------#

    Ctrl+n to create a new line for multiline prompts.
    Type exit to quit.


    '''))

bindings = KeyBindings()

@bindings.add('c-n')
def _(event):
    event.current_buffer.insert_text('\n')

@bindings.add('enter')
def _(event):
    event.current_buffer.validate_and_handle()


def input_prompt(session):
    print_formatted_text(HTML('<b>Prompt:</b>'))
    prompt = session.prompt(multiline=True, key_bindings=bindings)

    if prompt == 'exit':
        exit(0)
    if prompt == '':
        input_prompt(session)

    return prompt

def output_response(response):
    response = response.lstrip()

    print('\n')
    print_formatted_text(HTML('<b>GPT-3: </b>'))
    print(f"{response}")
    print('\n')

def set_api_key(api_key):
    if api_key == '':
        return gpt.get_api_key()
    else:
        return api_key


def main():
    args = parse_args()
    max_tokens = int(args.max_tokens)
    model = args.model
    temperature = float(args.temperature)
    api_key = args.api_key


    openai.api_key = set_api_key(api_key)

    if openai.api_key is None:
        print("Please set the OPENAI_API_KEY environment variable. Eg: export OPENAI_API_KEY=sk-XXXXXX. Or use the --api_key argument.")
        exit(1)


    welcome_message()

    session = PromptSession()

    prompt = ''
    # Chat loop
    while True:
        prompt = input_prompt(session)
        response = gpt.submit_prompt(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
        output_response(response)
    

if __name__ == "__main__":
    main()