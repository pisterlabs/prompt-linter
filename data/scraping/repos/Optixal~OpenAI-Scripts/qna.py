#!/usr/bin/env python3

import os
import sys
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')


def create(question):
    return openai.Completion.create(
        engine="davinci",
        prompt=
        f"Q: {question}\nA:",
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"])


def format_response(response):
    if 'choices' in response and len(response['choices']) >= 1:
        return response['choices'][0]['text'].strip()
    else:
        return 'No response'


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} "topic of interest"')
        exit(1)

    question = sys.argv[1]
    response = create(question)
    print(response)
    output = format_response(response)
    print(output)


if __name__ == '__main__':
    main()
