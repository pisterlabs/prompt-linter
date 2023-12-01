# Initialise tokenizer
import codecs
import os

import joblib
import openai
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

results = []

import openai
from config import *


# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j


def loadContent(filepath):
    filep = '%s' % (filepath)
    return codecs.open(filep, 'r', 'utf-8').read()


texts = {}
texts['1.txt'] = loadContent('../data/1.txt')
texts['2.txt'] = loadContent('../data/2.txt')
texts['3.txt'] = loadContent('../data/3.txt')
texts['4.txt'] = loadContent('../data/4.txt')
texts['5.txt'] = loadContent('../data/5.txt')

# 对每个段落做embeddings
text_embeddings = {}
for text_name, text_content in texts.items():
    response = openai.Embedding.create(
        input=text_content,
        engine="text-embedding-ada-002"
    )
    text_embeddings[text_name] = {'embedding': response['data'][0]['embedding'], 'content': text_content}

joblib.dump(text_embeddings, 'text_embeddings.pkl')

print("end")
