import sys
import json
import os
import re
import requests
import lxml.html
import openai


def get_article_summary(text, max_tokens):
    api_key = ''
    openai.api_key = api_key
    openai.organization = ''

    response = openai.Completion.create(
        engine="davinci",
        prompt=text,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"])

    return response

if __name__ == '__main__':
    filename = sys.argv[1]
    with open(filename) as f:
        data = json.load(f)

    output = []
    for article in data:
        output_text = []
        title = get_article_summary(article['title'][0].replace('| Southern Living', ''), 60)
        for text in article['text']:
            input_text = text + "\n\ntl;dr:"

            summary = get_article_summary(input_text, 90)
            output_text.append(summary['choices'][0]['text'])

        output.append({
            'source_url': article['source_url'],
            'source_text': article['text'],
            'zobot': output_text,
            'source_title': article['title'][0],
            'output_tile': title['choices'][0]['text'].replace('| Southern Living', '')
        })

    json.dump(output, sys.stdout, indent=4)

