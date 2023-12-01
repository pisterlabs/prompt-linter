import openai, pinecone
from datasets import load_dataset
import os 

from dotenv import load_dotenv
load_dotenv()
openai_key = os.environ.get("OPENAI_API_KEY")
pinecone_key = os.environ.get("PINECONE_API_KEY")
openai.api_key = openai_key
pinecone.init(api_key=pinecone_key,
             environment='us-west4-gcp-free')
index = pinecone.Index('hacktest')

MODEL = 'text-embedding-ada-002'
def query_vb(query):
    # create the query embedding
    xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']
    
    # query, returning the top 5 most similar results
    res = index.query([xq], top_k=2, include_metadata=True)
    return res

#### FOR INITIALIZING DATABASE
# dataset = load_dataset('json', data_files='fake_data.json')['train']
# MODEL ='text-embedding-ada-002'
# ids = [str(n) for n in range(len(dataset['remarks']))]
# input = dataset['remarks']
# res = openai.Embedding.create(input=input, engine=MODEL)
# embeds = [record['embedding'] for record in res['data']]
# meta = [{'text': text} for text in dataset['text']]
# to_upsert = zip(ids, embeds, meta)
# index.upsert(vectors=list(to_upsert))
