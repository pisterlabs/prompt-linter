import json
import requests
import urllib.request
from PIL import Image
import openai
import cohere

openai.api_key = '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'
cohere_key = 'MNhF0rXzQm56XN7OBlEgyGqMw0fbucDB406e9agC'

co = cohere.Client('MNhF0rXzQm56XN7OBlEgyGqMw0fbucDB406e9agC')  # This is your trial API key


def create_picture(prompt, k):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )
    image_url = response['data'][0]['url']
    urllib.request.urlretrieve(image_url, f"image_{k}.png")
    return f"image_{k}.png"


def analyze(k):
    f = open(f'research-papers/research-1.txt')
    for i in range(k - 1):
        f.readline()
    text = f.readline()
    response = co.summarize(
        text=text,
        length='medium',
        format='auto',
        model='command',
        additional_command='Explain this article to a 10 year old. ',
        temperature=0.2,
    )
    summary = response.summary

    response = co.summarize(
        text=text,
        length='short',
        format='auto',
        model='command',
        additional_command='Make it sound eye popping, groundbreaking news headline.',
        temperature=0.1,
    )
    headline = response.summary

    loc = create_picture(headline, k)
    return headline, summary, loc
