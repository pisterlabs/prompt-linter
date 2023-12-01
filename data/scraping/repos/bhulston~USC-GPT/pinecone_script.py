from langchain.embeddings import OpenAIEmbeddings  
from langchain.vectorstores import Pinecone
import pinecone
from bs4 import BeautifulSoup
import requests
import csv
import pandas as pd
import requests
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('course_data.csv')

pinecone.init(
    api_key="12aeea93-0da1-47a1-bb32-193bda02eadc",
    environment="gcp-starter" 
)

index_name = "use-class-db"

# if index_name not in pinecone.list_indexes():
#     # if does not exist, create index
#     pinecone.create_index(
#         index_name,
#         dimension = 1536,
#         metric='cosine'
#         # metadata_config={
#         #     'indexed': ['text', 'program', 'url', 'code', 'name', 'units', 'prerequisites', 'section', 'time', 'days', 'class_type']
#         # }
#     )

index = pinecone.GRPCIndex(index_name)
print("Index Retrieved:", index)


ids = []
embeds = []
metadatas = []
batch = 1
count = 0

embeddings = OpenAIEmbeddings(openai_api_key="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")


print('Building vectors')
for _, row in df.iterrows():
    if batch % 100 == 0:
        count += 1
        res = embeddings.embed_documents(embeds)
        texts = []
        for i in range(len(embeds)):
            texts.append(res[i])
        index.upsert(vectors = zip(ids, texts, metadatas))
        print('Upsert complete: Batch', count)

        ids = []
        embeds = []
        metadatas = []
       
    page_content = f"{row['Class ID']}: {row['Class Name']} is taught by {row['Instructor']}. The description of the class is: {row['Catalogue']}"
    embeds.append(page_content)
    metadata = {
        "text": page_content,
        'program': row['Program Code'],
        'url': row['Program URL'],
        'code': row['Class ID'],
        'name': row['Class Name'],
        'units': row['Units'],
        'prerequisites': row['Prerequisites'],
        'restrictions': row['Restrictions'],
        'section': row['Class Section'],
        'time': row['Time'],
        'days': row['Days'],
        'class_type': row['Class Type']
    }
    ids.append(str(batch))
    metadatas.append(metadata)
    batch += 1

if len(ids) > 0:
    res = embeddings.embed_documents(embeds)
    texts = []
    for i in range(len(embeds)):
        texts.append(res[i])
    index.upsert(vectors = zip(ids, texts, metadatas))

print(index.describe_index_stats())

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embeddings.embed_query, text_field
)
    
query = "Find me an AI class"

print(vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
))