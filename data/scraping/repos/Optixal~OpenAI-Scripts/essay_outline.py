#!/usr/bin/env python3

import os
import sys
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')


def create(topic):
    return openai.Completion.create(
        engine='davinci',
        prompt=
        f'Create an outline for an essay about {topic}:\n\nI: Introduction',
        temperature=0.7,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0)


def format_response(response):
    buffer = 'I: Introduction'
    if 'choices' in response and len(response['choices']) >= 1:
        buffer += response['choices'][0]['text']
        return buffer.strip()
    else:
        return 'No response'


def main():
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} "topic of interest"')
        exit(1)

    topic = sys.argv[1]
    response = create(topic)
    print(response)
    output = format_response(response)
    print(output)


if __name__ == '__main__':
    main()
