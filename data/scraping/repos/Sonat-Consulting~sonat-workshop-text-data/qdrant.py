import os
from langchain.vectorstores import Qdrant
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from pathlib import Path
from utils import get_url_from_filename, get_source

from langchain.embeddings.openai import OpenAIEmbeddings

directory_path = './data'

# List the directory contents using os.listdir
directory_contents = os.listdir(directory_path)
files = []
ids = []
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=100, separator = " ")
chunks = []
# Iterate through the directory contents
for item in directory_contents:
    # Create the full item path by joining the directory path and the item name
    item_path = os.path.join(directory_path, item)

    if ".txt" in item_path:
        source = get_source(item_path)

        with open(item_path) as f:
            file = f.read()
        
        text_split = text_splitter.split_text(file)
        text_documents = []
        for i in text_split:
            text_documents.append(Document(page_content=i, metadata=dict(path = get_url_from_filename(Path(item_path).name), source=source)))
        chunks.extend(text_documents)

qdrant = Qdrant.from_documents(
   chunks,
   OpenAIEmbeddings(),
   url="http://localhost:6333",
   prefer_grpc=True,
   collection_name="brreg"
)