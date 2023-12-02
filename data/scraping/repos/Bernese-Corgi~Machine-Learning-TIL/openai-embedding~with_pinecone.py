import os
import openai
import pandas as pd
import pinecone
from datasets import load_dataset
from tqdm.auto import tqdm


openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone_key = os.getenv('PINECONE_API_KEY')
print(openai.Engine.list())

data_array = list(pd.read_csv('data/blog_posts.csv')['content_text'].values)

MODEL = "text-embedding-ada-002"
res = openai.Embedding.create(
    input=data_array,
    engine=MODEL
)

embeds = [record['embedding'] for record in res['data']]

pinecone.init(
    api_key=pinecone_key,
    environment="asia-southeast1-gcp-free"
)

if 'openai' not in pinecone.list_indexes():
    pinecone.create_index('openai', dimension=len(embeds[0]))
    
index = pinecone.Index('openai')

trec = load_dataset('trec', split='train[:10]')

batch_size = 32

for i in tqdm(range(0, len(trec['text']), batch_size)):
    i_end = min(i + batch_size, len(trec['text']))
    lines_batch = trec['text'][i:(i + batch_size)]
    ids_batch = [str(n) for n in range(i, i_end)]
    res = openai.Embedding.create(input=lines_batch, engine=MODEL)
    embeds = [record['embedding'] for record in res['data']]
    meta = [{ 'text': line } for line in lines_batch]
    to_upsert = zip(ids_batch, embeds, meta)
    index.upsert(vectors=list(to_upsert))

query = "What caused the 1929 Great Depression?"

xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']

res = index.query([xq], top_k=5, include_metadata=True)

for match in res['matches']:
    print(f"{match['score']:.2f}:{match['metadata']['text']}")