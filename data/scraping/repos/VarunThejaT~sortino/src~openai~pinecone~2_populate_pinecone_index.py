import os
import pinecone
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL = "text-embedding-ada-002"

res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        # "there will be several phrases in each batch"
    ], engine=MODEL
)

# extract embeddings to a list
embeds = [record['embedding'] for record in res['data']]

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="northamerica-northeast1-gcp"  # find next to API key in console
)

# check if 'openai' index already exists (only create index if not)
if 'openai' not in pinecone.list_indexes():
    pinecone.create_index('openai', dimension=len(embeds[0]))
# connect to index
index = pinecone.Index('openai')

from datasets import load_dataset

# load the first 1K rows of the TREC dataset
trec = load_dataset('trec', split='train[:1000]')


from tqdm.auto import tqdm  # this is our progress bar

batch_size = 32  # process everything in batches of 32
# using batch_size instead of len(trec['text']) for testing purposes
for i in tqdm(range(0, len(trec['text']), batch_size)):
    # set end position of batch
    i_end = min(i+batch_size, len(trec['text']))
    # get batch of lines and IDs
    lines_batch = trec['text'][i: i+batch_size]
    ids_batch = [str(n) for n in range(i, i_end)]
    # create embeddings
    res = openai.Embedding.create(input=lines_batch, engine=MODEL)
    embeds = [record['embedding'] for record in res['data']]
    # prep metadata and upsert batch
    metadata = [{'text': line} for line in lines_batch]
    to_upsert = zip(ids_batch, embeds, metadata)
    # upsert to Pinecone
    index.upsert(vectors=list(to_upsert))

query = "What was the cause of the major recession in the early 20th century?"

# create the query embedding
xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']

# query, returning the top 5 most similar results
res = index.query([xq], top_k=5, include_metadata=True)

for match in res['matches']:
    print(f"{match['score']:.2f}: {match['metadata']['text']}")
