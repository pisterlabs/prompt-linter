import os
import sys
from uuid import uuid4
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, UnstructuredFileLoader, PyPDFLoader
import pinecone
from pdfminer.high_level import extract_text
from langchain.embeddings.openai import OpenAIEmbeddings
from src.constants import OPENAPI_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from tqdm.auto import tqdm
from unstructured.cleaners.core import clean_extra_whitespace

def intialize_dependencies():
    model_name = 'text-embedding-ada-002'
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT,
    )

    embed = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=OPENAPI_KEY
    )

    index = pinecone.Index(PINECONE_INDEX_NAME)
    vectorstore = Pinecone(index, embed.embed_query, "text")

    llm = ChatOpenAI(
    openai_api_key=OPENAPI_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.7
    )
    return (vectorstore, llm)

def query_embeddings(query, vectorstore, llm):

    vectorstore.similarity_search(
        query=query,
        k=10,
    )

    #qa = RetrievalQA.from_chain_type(
    #llm=llm,
    #chain_type="stuff",
    #retriever=vectorstore.as_retriever()
    #)
    #print(qa.run(query))
    qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    print(qa_with_sources(query))
def terminal_query(vectorstore, llm):
    print("Input a query (type exit to exit)")
    query = input()
    while query.lower() != 'exit':
        query_embeddings(query, vectorstore, llm)
        print("Input a query (type exit to exit)")
        query = input()

TEXT_PATH = "/Users/devmasrani/Documents/StudyBuddy/data/text_files"

def main(argv):
    VERBOSE = True  if "-v" in argv else False
    NEW_DATA = True if "-n" in argv else False
    SKIP_CONVERSION = True if "-s" in argv else False
    
    if NEW_DATA:
        #loader = DirectoryLoader(os.path.join(os.getcwd(), 'data'), show_progress=True, loader_cls=UnstructuredFileLoader, loader_kwargs={'mode': "single", 'post_processors': [clean_extra_whitespace]} )
        loader = DirectoryLoader(os.path.join(os.getcwd(), 'data'), show_progress=True, loader_cls=PyPDFLoader)
        
        data = loader.load() #Data is an an array of Document objects with each object having a page_content and metadata
        #print(data)
        initalize_embeddings(data)
    
    vectorstore, llm = intialize_dependencies()

    terminal_query(vectorstore, llm)


