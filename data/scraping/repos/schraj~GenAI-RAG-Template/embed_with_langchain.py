import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import (
    TextLoader,
)
from langchain.embeddings import OpenAIEmbeddings
from python_api.shared.app_base import initialize_openai

load_dotenv()

embeddings_model = os.environ.get('EMBEDDINGS_MODEL')
chunk_size = 500
chunk_overlap = 50

file_path = os.path.join('source_documents', "state_of_the_union.txt") 

def load_document() -> Document:
    loader_args = {"encoding": "utf8"}
    loader = TextLoader(file_path, **loader_args)
    return loader.load()
    
def chunk_document(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(document)
    return texts

def process_document() -> Document:
    document = load_document()
    return chunk_document(document)

def main():
    initialize_openai()
    
    texts = process_document()
    sample_text = texts[0].page_content
    embeddings = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        chunk_size=1)

    print(embeddings.embed_query(sample_text))

    print(f"Embeddings complete!")

if __name__ == "__main__":
    main() 

