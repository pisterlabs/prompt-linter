from decouple import config
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import os
from langchain.document_loaders.base import Document
from langchain.indexes import VectorstoreIndexCreator

from langchain.utilities import ApifyWrapper


# Load environment variables from a .env file using decouple
WEBSITE_URL = "https://theavesattwelve100.com/"

if __name__ == '__main__':
    apify = ApifyWrapper()
    loader = apify.call_actor(
        actor_id="apify/website-content-crawler",
        run_input={"startUrls": [{"url": "https://python.langchain.com/en/latest/"}]},
        dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
            ),
    )
 
    index = VectorstoreIndexCreator().from_loaders([loader])
    
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator='\n')
    docs = text_splitter.split_documents(documents)

    # Set the chunk_size parameter for OpenAIEmbeddings to 1
    embedding = OpenAIEmbeddings(chunk_size=1)

    persist_directory = 'db'
    vectordb = Chroma.from_documents(documents=docs, embeddings=embedding, persist_directory=persist_directory)
    vectordb.persist()
