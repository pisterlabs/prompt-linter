import glob
import os
import openai
import pinecone
import time

from uuid import uuid4
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter, Language

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv(), override=True) # read local .env file

openai.api_key = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'
pinecone_api_key = os.getenv('PINECONE_API_KEY') or 'YOUR_API_KEY'
pinecone_env = os.getenv('PINECONE_ENVIRONMENT') or "YOUR_ENV"

base_dir = "./files"


# LOADING

markdown_files = glob.glob(os.path.join(base_dir, '**/*.md'), recursive=True)
docs = [UnstructuredMarkdownLoader(f, mode = "single").load()[0] for f in markdown_files]

"""
 ____  _  _  ____  ____  ____  ____  __  __  ____  _  _  ____  ___ 
( ___)( \/ )(  _ \( ___)(  _ \(_  _)(  \/  )( ___)( \( )(_  _)/ __)
 )__)  )  (  )___/ )__)  )   / _)(_  )    (  )__)  )  (   )(  \__ \
(____)(_/\_)(__)  (____)(_)\_)(____)(_/\/\_)(____)(_)\_) (__) (___/
"""

"""

If you want to use MarkdownHeaderTextSplitter, use this code

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splitted_docs = [markdown_splitter.split_text(doc.page_content) for doc in docs]

If you want to use from_language, use this code

md_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
)

md_docs = [md_splitter.create_documents([markdown_text.page_content]) for markdown_text in docs]
"""

# SPLITTING
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap  = 128,
)

# IMPORTANT VARIABLE
splitted_docs = text_splitter.split_documents(docs)

# EMBEDDING
model_name = 'text-embedding-ada-002'
texts = [c.page_content for c in splitted_docs]

print("Created", len(texts), "texts")

chunks = [texts[i:(i + 1000) if (i+1000) <  len(texts) else len(texts)] for i in range(0, len(texts), 1000)]
embeds = []

print("Have", len(chunks), "chunks")
print("Last chunk has", len(chunks[-1]), "texts")

for chunk, i in zip(chunks, range(len(chunks))):
    print("Chunk", i, "of", len(chunk))
    new_embeddings = openai.Embedding.create(
        input=chunk,
        model=model_name,
    )
    new_embeds = [record['embedding'] for record in new_embeddings['data']]
    embeds.extend(new_embeds)

# PINECONE STORE
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

index_name = 'zkappumstad'

# Use this for just first time to upload

if index_name in pinecone.list_indexes():
    pinecone.delete_index(index_name)

pinecone.create_index(
    name=index_name,
    metric='dotproduct',
    dimension=1536
) 

while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pinecone.Index(index_name)

def extract_title(document):
    lines = document.page_content.split('\n')
    for line in lines:
        if line.startswith('title:'):
            title = line.split('title:')[1].strip()
            return title
        elif line.startswith('# '):
            title = line.split('#')[1].strip()
            return title
    return ""

ids = [str(uuid4()) for _ in range(len(splitted_docs))]

vector_type = os.getenv('DOCS_VECTOR_TYPE') or 'DOCS_VECTOR_TYPE'

vectors = [(ids[i], embeds[i], {
    'text': splitted_docs[i].page_content, 
    'title': extract_title(splitted_docs[i]),
    'vector_type': vector_type,
    }) for i in range(len(splitted_docs))]

for i in range(0, len(vectors), 100):
    batch = vectors[i:i+100]
    index.upsert(batch)

print(index.describe_index_stats())