import time

import openai
import pandas as pd
import os
import tiktoken
import numpy as np
import json

max_tokens = 1000


def read_and_preprocess_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def split_into_many(text, max_tokens=max_tokens):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    sentences = text.split('. ')

    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    for sentence, token in zip(sentences, n_tokens):
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        if token > max_tokens:
            continue

        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


def convert_to_openai(x):
    try:
        return openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding']
    except Exception as ex:
        print(ex)
        print("Sleeping")
        time.sleep(60)
        print("Continue")
        return convert_to_openai(x)


def process_from_file(outname):
    texts = read_and_preprocess_json("jailbreaks.json")

    df = pd.DataFrame(texts, columns=['name', 'value'])
    df.to_csv('scraped.csv')
    df.head()
    tokenizer = tiktoken.get_encoding("cl100k_base")

    shortened = []

    df['n_tokens'] = df.value.apply(lambda x: len(tokenizer.encode(x)))

    for row in df.iterrows():
        if row[1]['value'] is None:
            continue

        if row[1]['n_tokens'] > max_tokens:
            shortened += list(map(lambda x: (row[1]['name'], x), split_into_many(row[1]['value'])))

        else:
            shortened.append((row[1]['name'], row[1]['value']))

    df = pd.DataFrame(shortened, columns=['name', 'value'])

    df['n_tokens'] = df.value.apply(lambda x: len(tokenizer.encode(x)))

    df['embeddings'] = df.value.apply(
        lambda x: convert_to_openai(x))

    df.to_csv(outname)
    df.head()


def setup_openai():
    openai.api_key = os.environ.get('OPENAI_API_KEY')


if __name__ == "__main__":
    setup_openai()
    process_from_file("scanned.csv")
