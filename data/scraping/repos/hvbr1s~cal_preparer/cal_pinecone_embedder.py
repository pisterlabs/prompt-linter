import hashlib
import json
import os
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import pinecone
import tiktoken
from tqdm.auto import tqdm
import os
from bs4 import BeautifulSoup
from pathlib import Path
from llama_index import download_loader

class UnstructuredFileLoader:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def load(self):
        docs = []
        for file_name in os.listdir(self.directory_path):
            file_path = os.path.join(self.directory_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith('.html'):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                url = extract_url_from_html(file_path)
                if url:
                    metadata = {"source": url}
                else:
                    metadata = {}

                docs.append(Document(page_content=content, metadata=metadata))
        return docs

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def extract_url_from_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    url_meta_tag = soup.find("meta", attrs={"name": "url"})
    if url_meta_tag:
        return url_meta_tag.get("content")
    return None

load_dotenv()

env_vars = [
    'OPENAI_API_KEY',
    'PINECONE_API_KEY',
    'PINECONE_ENVIRONMENT'
]
os.environ.update({key: os.getenv(key) for key in env_vars})

openai.api_key = os.getenv('OPENAI_API_KEY')
embed_model = "text-embedding-ada-002"

################################## DATA PREP ######################################


# Initialize the loader and load documents
loader = UnstructuredFileLoader("/home/dan/zendesk_backup/cal/")
docs = loader.load()
print(len(docs))

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding('cl100k_base')

# Initialize the MD5 hash object
m = hashlib.md5()

# Initialize the chunk list
documents = []

# Define the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=10,  # number of tokens overlap between chunks
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
)


# Process each document
for doc in tqdm(docs):
    #m.update(url.encode('utf-8'))
    uid = m.hexdigest()[:12]
    chunks = text_splitter.split_text(doc.page_content)
    print(len(chunks))
    
    # Create an entry for each chunk
    for i, chunk in enumerate(chunks):
        # Remove HTML tags from the chunk before appending
        text_without_tags = BeautifulSoup(chunk, "html.parser").get_text()
        documents.append({
            'id': f'{uid}-{i}',
            'text': text_without_tags,
            'chunk': i,
        })

texts = [doc['text'] for doc in documents]

# Print the total number of documents
print(len(documents))


################################## EMBEDDING ######################################

res = openai.Embedding.create(
    input=texts, 
    engine=embed_model
)

index_name = 'cal'

# initialize connection to pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT')
)

# check if index already exists
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='dotproduct'
    )
# connect to index
index = pinecone.GRPCIndex(index_name)
# view index stats


batch_size = 100  # how many embeddings we create and insert at once

for i in tqdm(range(0, len(documents), batch_size)):
    # find end of batch
    i_end = min(len(documents), i+batch_size)
    meta_batch = documents[i:i_end]
    # get ids
    ids_batch = [x['id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text'] for x in meta_batch]
    # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                pass
    embeds = [record['embedding'] for record in res['data']]
    # cleanup metadata
    meta_batch = [{
        'text': x['text'],
        'chunk': x['chunk'],
    } for x in meta_batch]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)
