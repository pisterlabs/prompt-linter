# pylint: disable=line-too-long
'''
This module will create the methods to generate branding snippet and keywords by run GPT-3 model using text-davinci-002
'''
import argparse
import re
import openai
from decouple import config
from typing import List


MAX_INPUT_LENGTH = 12


def main():
    '''
    >>> This method will run GPT-3 model to generate branding snippet and keywords
    '''
    print('Running GPT-3 model...\n')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)

    args = parser.parse_args()
    user_input = args.input

    if validate_length(user_input):
        branding_result = generate_branding_snippet(user_input)
        keywords_result = generate_keywords(user_input)
        print(branding_result)
        print(keywords_result)
    else:
        raise ValueError(
            f'Input length is too long. Must be under {MAX_INPUT_LENGTH}. Submitted input is {len(user_input)}')


def validate_length(prompt: str) -> bool:
    '''
    >>> This method validate if prompt has a lenght equal or less then MAX_INPUT_LENGTH
    '''
    return len(prompt) <= MAX_INPUT_LENGTH


def generate_branding_snippet(prompt: str) -> str:
    '''
    >>> This method generate the brading snippet by passing a prompt argument in GPT-3 model
    '''
    openai.api_key = config('OPENAI_API_KEY')

    enriched_prompt = f'Generate upbeat branding snippet for {prompt}'

    response = openai.Completion.create(
        engine="text-davinci-002", prompt=enriched_prompt, max_tokens=32)

    branding_text = response['choices'][0]['text'].strip()
    last_char = branding_text[-1]

    if last_char not in ('.', '!', '?'):
        branding_text += '...'

    return branding_text


def generate_keywords(prompt: str) -> List[str]:
    '''
    >>> This method generate the related branding keywords by passing a prompt argument in GPT-3 model
    '''
    openai.api_key = config('OPENAI_API_KEY')

    enriched_prompt = f'Generate related branding keywords for {prompt}'

    response = openai.Completion.create(
        engine="text-davinci-002", prompt=enriched_prompt, max_tokens=32)

    keywords_text = response['choices'][0]['text'].strip()

    keywords_array = re.split(',|\n|;|-', keywords_text)
    keywords_array = [key.lower().strip() for key in keywords_array]
    keywords_array = [key for key in keywords_array if len(key) > 0]

    return keywords_array


if __name__ == '__main__':
    main()
