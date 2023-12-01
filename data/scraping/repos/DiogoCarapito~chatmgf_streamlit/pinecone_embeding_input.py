import pandas as pd
import openai
import pinecone
from dotenv import load_dotenv
import os
from tqdm.auto import tqdm
import datetime
from time import sleep

load_dotenv()  # Load variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
print('api loaded')

# read csv file
df = pd.read_csv('scrapped_data/Recomendações_choosing_wisely_30-03-2023.csv')
print('csv file read')

#print(df[0:4])
text_for_embedding = []
for index, row in df[0:4].iterrows():
    text_for_embedding.append(row['Recomendação'])

print('text for embedding prepared')
#print(text_for_embedding)

# openai api key
openai.api_key = OPENAI_API_KEY
embed_model = "text-embedding-ada-002"

print('OpenAI initialized')

'''# embeding texts
res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=embed_model
)'''

index_name = 'choosing-wisely-test'

# initialize connection to pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # app.pinecone.io (console)
    environment=PINECONE_API_ENV  # next to API key in console
)
print('Pinecone initialized')

# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='dotproduct'
    )
    print('pinecone index created')


# connect to index
#index = pinecone.GRPCIndex(index_name)
index = pinecone.Index(index_name)
print('pinecone index connected')


to_upsert=[]
vectors = []
ids = []


#for texts in text_for_embedding:
for idx, row in df[0:4].iterrows():
    texts = row['Recomendação']

    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
        print('embedding created')
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                print('embedding created')
                done = True
            except:
                print('embedding NOT created')
                pass

    embeds = [record['embedding'] for record in res['data']]

    tuple_to_upsert = (str(index), embeds[0])
    to_upsert.append(tuple_to_upsert)
    #vectors.append(embeds[0])
    #ids.append(idx)
    #print(index)
    #print(embeds[0])

print(to_upsert)
#print(vectors)
#print(ids)

# upsert to Pinecone
index.upsert(vectors=to_upsert)
#index.upsert(vectors=vectors, ids=ids)



#print(index.describe_index_stats())
# check if upsert was successful

index_description = index.describe_index_stats()
print(index_description)
print('###')
if index_description['total_vector_count'] > 0:
    print('upserting to pinecone successful')
else:
    print('upserting to pinecone NOT successful')

