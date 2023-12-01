import time
import os
import pinecone
from datasets import load_dataset
from tqdm.auto import tqdm
import datetime
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from uuid import uuid4

load_dotenv()
MODEL = "text-embedding-ada-002"
embed = OpenAIEmbeddings(
    model=MODEL, openai_api_key=os.getenv("OPENAI_API_KEY"))
res = embed.embed_documents(
    ['this is the first chunk of text', 'then another second chunk of text is here'])

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    # find next to api key in console
    environment=os.getenv("PINECONE_ENVIRONMENT")
)
# embeds = [record['embedding'] for record in res['data']]
# check if 'openai' index already exists (only create index if not)
# embeds = [record['embedding'] for record in res['data']]
index_name = os.getenv("PINECONE_INDEX_NAME")
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=len(res[0]))
# connect to index
index = pinecone.Index(index_name)

# load the entire TREC dataset
poultry = load_dataset('csv', data_files='embedding_text.csv')['train']

print(len(poultry['embedding']))

count = 0  # we'll use the count to create unique IDs
batch_size = 300  # process everything in batches of 32
for i in tqdm(range(0, len(poultry['embedding']), batch_size)):
    while True:
        try:
            current_time = datetime.datetime.now().time()
            # set end position of batch
            i_end = min(i+batch_size, len(poultry['embedding']))
            # get batch of lines and IDs
            lines_batch = poultry['embedding'][i: i_end]
            # print(lines_batch)
            source_batch = poultry['source'][i: i_end]
            ids_batch = [str(uuid4()) for _ in range(i, i_end)]
            # create embeddings
            embeds = embed.embed_documents(lines_batch)
            # prep metadata and upsert batch
            meta = [{'source': poultry['source'][n], 'chunk': n, 'text': poultry['embedding'][n]}
                    for n in range(i, i_end)]
            # print(ids_batch)
            # print(embeds)
            # print(meta)
            # break
            to_upsert = zip(ids_batch, embeds, meta)
            # upsert to Pinecone
            index.upsert(vectors=list(to_upsert),
                         namespace=os.getenv("PINECONE_NAMESPACE"))
            print(f"{i} to {i_end} finished successfully")
            break
        except:
            time.sleep(1)
            continue
