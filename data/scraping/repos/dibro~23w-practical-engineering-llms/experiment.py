import os
import re
import getpass
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from typing import List, Union

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI


os.environ["OPENAI_API_KEY"] = getpass.getpass("Your OpenAI API Key:")


def split_text(documents: List) -> List[langchain.schema.document.Document]:
    """ Two types of split: 
    - CharacterTextSplitter: split based on characters passed in, \n
    chunk size is amount of characters
    - RecursiveTextSplitter: split based on semantical units (e.g sentences) \n
    chunk size is numberd of characters or tokens -> We use this 
    """
    print("Splitting process...")
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len)
    
    chunks = text_split.transform_documents(documents)
    print('Splitting done')
    return chunks

def doc_embedding(chunks: List, emb_type) -> langchain.embeddings.cache.CacheBackedEmbeddings:
    """
    Construct an Embedder (get embeddings from chunks) with
    - CacheBackedEmbeddings: local cache
    - Some types of embeddings (emb_type): OpenAI, HuggingFace, etc
    """
    print("Constructing Embedder...")
    store = LocalFileStore("./cache/")
    # Choose embedding type
    if emb_type = 'openai':
        emb_model = OpenAIEmbeddings()
    if emb_type = 'huggingface':
        emb_model = HuggingFaceEmbeddings()
    else:
        raise ValueError(f'Unsupported embedding type: {emb_type}')
    
    embedder = CacheBackedEmbeddings(
        emb_model,
        store,
        namespace=emb_model.model
    )
    print('Embedder is ready!')
    return embedder


def vector_store(chunks: List[langchain.schema.document.Document],
                 embedder: langchain.embedding.cache.CacheBackedEmbeddings) -> langchain.vectorstores.faiss.FAISS:
    """
    Using embedder to transform chunks into vectors,
    then using FAISS to store them
    """
    print('Creating vector store...')
    vector_store = FAISS.from_documents(chunks, embedder)
    return vector_store

