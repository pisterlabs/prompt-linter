import os
import openai
import pinecone
import time

from uuid import uuid4
from langchain.document_loaders import GitbookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(), override=True) # read local .env file

openai.api_key = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'
pinecone_api_key = os.getenv('PINECONE_API_KEY') or 'YOUR_API_KEY'
pinecone_env = os.getenv('PINECONE_ENVIRONMENT') or "YOUR_ENV"


gitbook_path = "https://docs.aurowallet.com"

# LOADING
loader = GitbookLoader(gitbook_path, load_all_paths=True)
all_pages_data = loader.load()

print(f"fetched {len(all_pages_data)} documents.")

# SPLITTING
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 128,
)

splitted_docs = text_splitter.split_documents(all_pages_data)

# EMBEDDING
model_name = 'text-embedding-ada-002'
texts = [c.page_content for c in splitted_docs]

embeddings = openai.Embedding.create(
    input=texts,
    model=model_name,
)

# IMPORTANT VARIABLE
embeds = [record['embedding'] for record in embeddings['data']]

# PINECONE STORE
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

index_name = 'zkappumstad'
index = pinecone.Index(index_name)

def extract_title(document):
    lines = document.page_content.split('\n')
    for line in lines:
        if line.startswith('title:'):
            title = line.split('title:')[1].strip()
            return title
    return ""

ids = [str(uuid4()) for _ in range(len(splitted_docs))]

vector_type = os.getenv('DOCS_VECTOR_TYPE') or 'DOCS_VECTOR_TYPE'

vectors = [(ids[i], embeds[i], {
    'text': splitted_docs[i].page_content, 
    'title': extract_title(splitted_docs[i]),
    'vector_type': vector_type,
    }) for i in range(len(splitted_docs))]


"""
if vectors has more than 100 elements, use batch upsert with loop
"""
print(vectors[0])
print(index.describe_index_stats())

index.upsert(vectors)

print(f"upserted {len(vectors)} documents")
time.sleep(5)
print(index.describe_index_stats())