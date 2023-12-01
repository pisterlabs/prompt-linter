import os
import pandas as pd
import tiktoken
import openai

openai.api_key_path = 'apikey.txt'
data = {'name': [], 'tokens': [], 'text': []}
for filename in os.listdir('text/www.centific.com'):
    with open(f'text/www.centific.com/{filename}', 'r', encoding='UTF-8') as f:
        data['name'].append(filename[:-4])
        text = f.read()
        data['text'].append(text)
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        num_tokens = len(encoding.encode(text))
        data['tokens'].append(num_tokens)

df = pd.DataFrame(data)
df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
df.to_csv('text/www.centific.com.csv')
