## Lets see
## we want to read in the PDF x
## Maybe see if it has sections x
## Extract the section for Abastract and the Conclusions x
# ## Then do some research on vecotrizatiion
## We will use GROBID and this scipy_pdf package and see whats going on x

import scipdf
import glob
import openai
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

COLLECTION_NAME = 'pdf_search'
DIMENSION = 1536
connections.connect("default", host="host.docker.internal", port="19530")

# Remove collection if it already exists
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

# Create collection which includes the id, title, and embedding.
fields = [
    FieldSchema(name="id",dtype=DataType.INT64,is_primary=True, auto_id=True),
    FieldSchema(name='title', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='abstract', dtype=DataType.VARCHAR, max_length=64000),
    # FieldSchema(name='conclusion', dtype=DataType.VARCHAR, max_length=64000),
    FieldSchema(name='authors', dtype=DataType.VARCHAR, max_length=64000),
    # FieldSchema(name='pub_date', dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]
schema = CollectionSchema(fields=fields)
collection = Collection(name=COLLECTION_NAME, schema=schema)
INDEX_PARAM = {
    'metric_type':'L2',
    'index_type':"HNSW",
    'params':{'M': 8, 'efConstruction': 64}
}
BATCH_SIZE = 1000
# Create the index on the collection and load it.
collection.create_index(field_name="embedding", index_params=INDEX_PARAM)
collection.load()

## Setup ENV
import os
from dotenv import load_dotenv

load_dotenv()

GPT_3_API_KEY = os.getenv('GPT_3_API_KEY')
GPT_4_API_KEY = os.getenv('GPT_4_API_KEY')
GPT_4_ORG = os.getenv('GPT_4_ORG')

def processArticle(path):
    try:
        return scipdf.parse_pdf_to_dict(path, grobid_url="http://host.docker.internal:8071") # return dictionary
    except:
        print(f"Failed to process {path}")
        return None

def glob_folder(path): 
    files = glob.glob(path)
    return files

def collect_publish_data(globbed_files):
    # Goal:
    # Loop over glob of files
    # Post each on to the grobid url
    # Extract Abstract
    # Extract Author, Title, Reference List, Publish Date
    # Properties from GROBID
    # Title -> File Name
    # Authors -> delimited authors
    # pub_date -> Might be empty 
    # abstract -> text abstract
    # sections -> Array of Section ## May need to pull out conclusion from this array
    #   section: {
    #       heading: Section Heading,
    #       text: Text for section 
    #   }
    # references: Array of reference objects
    #   reference: {
    #       title: publication title
    #       journal: publication journal
    #       year: publish date
    #       authors: reference authors
    # }
    data = []
    for file in globbed_files: ## TODO: use asyncio to push this into parallel runs to make this faster
        processed_article = processArticle(file)
        if(processed_article == None or processed_article['abstract'] == ''):
            continue
        prepared_for_embedding = {
            'title': processed_article['title'] if processed_article['title'] != "" else file.split("/")[-1],
            'authors': processed_article['authors'] if processed_article['authors'] != "" else "Unknown",
            'pub_date': processed_article['pub_date'],
            'abstract': processed_article['abstract'],
            'conclusion': check_for_conclusion(processed_article)
        }
        data.append(prepared_for_embedding)
    return data

def check_for_conclusion(article):
    check_conclusion = [x for x in article['sections'] if 'conclusion' in x['heading'].lower() or 'discussion' in x['heading'].lower() or 'results' in x['heading'].lower()]
    if len(check_conclusion) > 0:
        return check_conclusion[0]['text']
    return ""

def create_conclusion_embedding(data):
    # Here we want to leverage the grobid breakdown to find the conclusion section and upload it.
    # Reach out to ChatGPT to get the embedding
    embed_model = "text-embedding-ada-002"
    results = []
    for item in data:
        if data['conclusion'] == "" or data['conclusion'] == None:
            continue
        embedding_result = openai.Embedding.create(
            input=[
            item['title']+";conclusion;"+item['conclusion'],
        ], engine=embed_model
        )
        results.append(embedding_result['data'][2]['embedding'])
    return


def create_abstract_embedding(data):
    # Reach out to ChatGPT to get the embedding
    embed_model = "text-embedding-ada-002"
    results = []
    for item in data:
        embedding_result = openai.Embedding.create(
            input=[
            item['title'],
            item['authors'],
            item['title']+";abstract;"+item['abstract'],
        ], engine=embed_model
        )
        results.append(embedding_result['data'][2]['embedding'])
    return results

def push_conclusion_into_milvus(data_points, embeddings):
    for i in range(0,len(embeddings)):
        data_points[i]['embedding'] = embeddings[i]
    entities = [
        [x['embedding'] for x in data_points]
    ]
    collection.insert(entities)
    return

def push_abstract_into_milvus(data_points, embeddings):
    for i in range(0,len(embeddings)):
        data_points[i]['embedding'] = embeddings[i]
    entities = [
        [x['title'] for x in data_points],
        [x['authors'] for x in data_points],
        [x['title']+x['abstract'] for x in data_points],
        [x['embedding'] for x in data_points]
    ]
    collection.insert(entities)
    return

openai.organization = GPT_4_ORG
openai.api_key = GPT_4_API_KEY
openai.Model.list()

import string

alpha = list(string.ascii_uppercase)
for item in alpha[10:]:
    path = f"./PDFs/{item}/*" ## TODO: set this as an input parameter to help streamline the processing
    files = glob_folder(path)
    processed_data = collect_publish_data(files)
    embeddings = create_abstract_embedding(processed_data)
    push_abstract_into_milvus(processed_data,embeddings)
    # embeddings = create_data_embedding(processed_data)
    # push_into_milvus(processed_data, embeddings)