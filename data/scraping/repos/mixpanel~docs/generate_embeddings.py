import os
import glob
import json
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity


openai.api_key = ''

def get_md_files(path):
    return [f for root, _, files in os.walk(path) for f in glob.glob(os.path.join(root, '*.md'))]

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def chunk_file(content, max_chunk_size=3000):
    return [content[i:i + max_chunk_size] for i in range(0, len(content), max_chunk_size)]

if __name__ == '__main__':
    with open('embeddings.json', 'r', encoding='utf-8') as f:
        embeddings = json.load(f)

    for f in get_md_files('.'):
        content = read_file(f)
        chunks = chunk_file(content)
        for i, c in enumerate(chunks):
            k = '{}-{}'.format(f, i)
            if k not in embeddings:
                try:
                    embeddings[k] = get_embedding(c)
                except Exception as e:
                    print(e)
                    print(f)
                    continue
                with open('embeddings.json', 'w', encoding='utf-8') as out:
                    json.dump(embeddings, out)
