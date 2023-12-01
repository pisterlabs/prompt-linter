from  langchain.schema import Document

import json
from typing import Iterable

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

docs=load_docs_from_jsonl('data.jsonl')

# Text Splitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs= text_splitter.split_documents(docs)



# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()


from langchain.vectorstores import Chroma
docsearch = Chroma.from_documents(docs, embeddings,persist_directory="chroma_db")