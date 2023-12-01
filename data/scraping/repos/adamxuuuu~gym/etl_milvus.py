from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Milvus
import os

from config import (
    COLLECTION_NAME,
    DIMENSION,
    MILVUS_HOST,
    MILVUS_PORT,
    EMBEDDING_MODEL
)

def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields=fields, description='search text')
    collection = Collection(name=collection_name, schema=schema)
    
    index_params = {
        'metric_type': "L2",
        'index_type': "IVF_FLAT",
        'params': {"nlist": 2048}
    }
    collection.create_index(field_name='embedding', index_params=index_params)
    utility.index_building_progress(COLLECTION_NAME)
    return collection

# Connect and Create to Milvus Database
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
collection = create_milvus_collection(COLLECTION_NAME, DIMENSION)

# Batch insert data
titles = []
contents = []
embeds = []
filedir = "data/"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# Get all the text files in the text directory
for file in os.listdir(filedir):
    txt = TextLoader(filedir + file).load()
    for c in text_splitter.split_documents(txt):
        titles.append(c.metadata['source'])
        contents.append(c.page_content)

data = [
    [i for i in range(len(contents))],
    titles,
    contents,
    embeddings.embed_documents(contents)
]
collection.insert(data)

collection.flush()
print(collection.num_entities)

