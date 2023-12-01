import io
import os
import sys

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from numpy import random
import requests
from sentence_transformers import SentenceTransformer

sys.path.append('../')
from api import db

os.environ['HUGGINGFACEHUB_API_TOKEN'] = ''
CHARTS_API_SUMMARY = 'https://charts.ecmwf.int/opencharts-api/v1/search/'

all_charts = requests.get(CHARTS_API_SUMMARY)
model = SentenceTransformer('all-mpnet-base-v2', device='mps')
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

if all_charts.status_code == 200:
    all_charts_json = all_charts.json()
    charts_specs = all_charts_json['results']

    for chart in charts_specs:
        openapi_json_schema = chart['schema-url']
        technical_name = chart['name']
        title = chart['title']
        description = chart['description']
        with open('temp.txt', 'w') as ftemp:
            ftemp.write(description)
        loader = TextLoader('temp.txt', encoding='utf-8')

        text = loader.load_and_split()
        documents = text_splitter.split_documents(text)
        for el in documents:
            embedding = {'id': random.randint(50000, 100000),
                         'vector': model.encode(el.page_content).tolist(),
                         'metadata': {"schema": openapi_json_schema,
                                      'name': technical_name,
                                      'title': title},
                         'text': el.page_content}
            db.db.ds.text.append(embedding['text'])
            db.db.ds.metadata.append(embedding['metadata'])
            db.db.ds.ids.append(embedding['id'])
            db.db.ds.embedding.append(embedding['vector'])
print("Done, committing to ds now!")
db.db.sd.commit()
        
