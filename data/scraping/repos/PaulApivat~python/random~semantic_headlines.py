import os
import json
import openai 
import chromadb

# use python-dotenv to get API key
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")


def get_embedding(text, model='text-embedding-ada-002'):
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

# indicate json file headlines are stored in.
with open('headlines.json', 'r') as file:
    all_headlines = json.load(file)

embeddings = []
for headline in all_headlines:
    embedding = get_embedding(headline['title'])
    embeddings.append(embedding)


# setup vector database
client = chromadb.Client()
collection = client.create_collection(name="all_headlines")


ids = [str(a['id']) for a in all_headlines] #chroma requires ids to be strings
documents = [a['title'] for a in all_headlines]
metadata = [{'topic': a['topic']} for a in all_headlines]
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadata,
    embeddings=embeddings
)

"""
Queries to try:

- Which headlines suggests things are going great in web3.
- I want to convince my artist friend to get into crypto.
- I need to testify in congress.
- I'm looking for a job, what stories should I read?
- I'm a protocol researcher, which stories should I read?
- I'm a day trader, what stories would be most interesting for me?

"""
query = "I'm a day trader, what stories would be most interesting for me?"
embedding = get_embedding(query)
results = collection.query(query_embeddings=[embedding], n_results=5)

print(results)