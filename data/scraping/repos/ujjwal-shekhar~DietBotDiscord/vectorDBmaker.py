import os
import platform

from dotenv import load_dotenv
load_dotenv()

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import UnstructuredURLLoader

"""
Load the data from the URLs provided.

The URLs that we have used were taken from the 
[ClearCals Blogs Section](https://clearcals.com/blogs).

We use the LangChain `UnstructuredURLLoader` to load the data
using the list of URLs provided. 

The function returns the data using the `load` method.
"""
def load_data_at_url():
    # Load the document
    urls = os.getenv('CONTEXT_URLS').split(',')
    loader = UnstructuredURLLoader(urls)
    data = loader.load()
    return data

"""
Convert the document into a vector database.

We use the `Chroma` vector store provided by LangChain to convert
the document into a vector database. We use the `OpenAIEmbeddings`
provided by LangChain to create the embedding. The `TokenTextSplitter`
provided by LangChain is used to split the document into chunks.
The `persist_directory` is used to store the vector database.

The function returns the vector database.
"""
def doc_to_vectordb(document):
    # Persist directory
    if platform.system() == 'Windows':
        persist_dir = os.path.join(os.environ['USERPROFILE'], 'chromadb')
    else:
        persist_dir = os.path.join(os.environ['HOME'], 'chromadb')

    # Create the embedding
    splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_doc = splitter.split_documents(document)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(split_doc, embeddings, 
                                     persist_directory=persist_dir)
    vectordb.persist()

    return vectordb
