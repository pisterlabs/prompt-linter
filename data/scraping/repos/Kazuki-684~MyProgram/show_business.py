import os

import openai
import chromadb

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, PDFMinerLoader

from django.http import HttpResponse
import logging

avoid_embedding = "on"

def make_show_business(pdf_path):
    
    #For spliting a text
    loader = PyPDFLoader(f"{pdf_path}")
    #MinerLoaderめちゃ重くない？？
    #loader = PDFMinerLoader(f"{pdf_path}")
    #text_spliter = CharacterTextSplitter(chunk_size=400)
    pages = loader.load_and_split()
    
    #OpenAI's apikey is already set to load from environment variables.
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    openai_ef = OpenAIEmbeddings()
    persist_path = "/home/kazuki/study/Study/4/Chroma"
    
    if (avoid_embedding != "on"):
        logging.debug("debug off")
        #Note that if a collection with the same collection name already exists, a new one will be created separately.
        collection = Chroma(collection_name="langchain2", embedding_function=openai_ef, persist_directory=persist_path)
        vectorstore = collection.from_documents(pages,embedding=openai_ef, persist_directory=persist_path)
    else:
        client = chromadb.PersistentClient(path=persist_path)
        vectorstore = Chroma(collection_name="langchain", client=client, embedding_function=openai_ef)
    
    pdf_qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

    query = "What is the nature of NTT DATA's business?"
    chat_history = []

    result = pdf_qa({"question": query, "chat_history": chat_history})
    
    return result["answer"]
    
    #return pages[0]