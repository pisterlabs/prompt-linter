
import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup

import pinecone

from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    GPTVectorStoreIndex,
    QuestionAnswerPrompt,
    PineconeReader
)

from llama_index.vector_stores import PineconeVectorStore
from llama_index.storage.storage_context import StorageContext
from langchain.chat_models import ChatOpenAI


from langchain.text_splitter import RecursiveCharacterTextSplitter
def chunk_text(text, chunk_size=700):  # chunk_size can be adjusted
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

import json
from langchain.document_loaders import JSONLoader
if __name__ == "__main__":
    loader = JSONLoader(
        file_path='blogs.txt',
        jq_schema='[].text',
        text_content=True)

    data = loader.load()
    processed_articles = []
    with open('blogs.txt', 'r') as f:
        articles = json.load(f)
        pass

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    # texts = text_splitter.split_documents(result)
    for article in articles:
        chunks = chunk_text(article['text'])
        for idx, chunk in enumerate(chunks):
            unique_id = f"{article['link']}_{idx}"
            
            # Process the chunk through Langchain (pseudo-code)
            # processed_chunk = langchain.process(chunk)
            
            processed_articles.append({
                'id': unique_id,
                'title': article['title'],
                'link': article['link'],
                'chunk': chunk,
                # 'processed_chunk': processed_chunk
            })


#             # Create a new document 
# doc = Document(page_content="Hello world!", 
#                metadata={"title": "My New Document"})
