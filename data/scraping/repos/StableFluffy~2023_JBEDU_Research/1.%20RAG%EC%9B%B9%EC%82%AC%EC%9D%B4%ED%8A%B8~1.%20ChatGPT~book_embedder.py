import os
import json
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
from chromadb.config import Settings
import chromadb
from uuid import uuid1

#config.json 불러오기
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "db/")
os.environ["OPENAI_API_KEY"] = CONFIG['openai_api_key']
openai.api_key = os.environ["OPENAI_API_KEY"]

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

client = chromadb.Client(
    Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=DB_DIR,
    )
)

folder_path = './pdf_standby'

file_names = [f[:-4] for f in os.listdir(folder_path) if not f.startswith('.DS_Store')]

embedding_function = OpenAIEmbeddings()
vectordb = client.get_or_create_collection('book', embedding_function=embedding_function)

print(file_names)

for file_name in file_names:
    with open(f'./pdf_standby/{file_name}.txt', 'r') as f: 
        text = f.read()

    print(text, file_name)
    print(f'./pdf_standby/{file_name}.txt')
    vectordb.add(ids=[str(uuid1())], embeddings=[get_embedding(text)], documents=[text], metadatas=[{"title": f"{file_name}"}])
    vectordb.peek()

client.persist()
vectordb.peek()