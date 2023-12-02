"""
pip install python-dotenv, openai[datalib]
"""

from dotenv import load_dotenv
load_dotenv()

import openai  # for generating embeddings
import os
import json

EMBEDDING_MODEL = "text-embedding-ada-002"
# embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
# max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191

openai.api_key = os.environ['OPENAI_KEY']

with open('./tmp/data/tag-posts-cat.json', 'r') as f:
    tag_posts = json.loads(f.read())
    tags = []
    text = []
    for tag, prop in tag_posts.items():
        to_embed = f'Tag:{tag}' + prop['text']
        tags.append(tag)
        text.append(to_embed)
    print('Embedding...')
    response = openai.Embedding.create(model=EMBEDDING_MODEL, input=text)
    print('Received...')
    out = []
    for i, (tag, resp) in enumerate(zip(tags, response["data"])):
        assert i == resp["index"]  # double check embeddings are in same order as input
        out.append((tag, resp['embedding']))
    with open('./tmp/data/tag-posts-embeddings.json', 'w') as fout:
        json.dump(out, fout)
