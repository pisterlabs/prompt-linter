from db.session import get_db
from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy.orm import Session
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from chromadb.config import Settings
import chromadb
from typing import List
import os

from langchain.document_loaders import SeleniumURLLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import SitemapLoader

router = APIRouter()

api = os.getenv("OPENAI_API_KEY")
#getting openapi key from env variable

@router.get("/query={query}")
def query(query: str, db: Session = Depends(get_db)):
    client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="backend/docs"))  
    # creating a new chroma db client which is persisted in backend/docs

    collection = client.get_or_create_collection("data")
    # creating/getting a new collection in the client called data

    res = collection.query(query_texts=[query] ,n_results=2)  
    # querying the collection for the query text and getting 2 results
    # res is a list of tuples of the form (id, score)
    # id is the id of the document in the collection
    # score is the score of the document with respect to the query
    
    chat = ChatOpenAI(openai_api_key=api
                      , model="gpt-3.5-turbo-0613")
    messages = [HumanMessage(content= "context : " + res + " question : " + query)]
    # providing the context and the question to the chatbot

    ans = chat(messages).content

    return ans
    
# function to add new documents to the collection
def add_doc(docs, index_num ):
    client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="backend/docs" ))
    collection = client.get_or_create_collection("data") 
    ids = list(range(index_num, index_num+len(docs)))
    charid = ["data" + str(i) for i in ids]
    collection.add(documents=docs,ids=charid)
    index_num = index_num+len(docs)
   
    print(collection.count())
    client.persist()
    
    return index_num

# endpoint to fetch text from url and add to collection
@router.post("/urlmethod/")
def train_urls(urls: List[str] ,index_num: int):
    loaders = SeleniumURLLoader(urls=urls)
    data = loaders.load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    for i in docs:
        add_doc([i.page_content], index_num)
    return "Done"

# endpoint to directly add text to collection
@router.post("/textmethod/")
def train_text(text: str ,index_num: int): 
    add_doc([text], index_num)
    return "Done"

# endpoint to fetch text from sitemap and add to collection
# sitemap is a xml file which contains the urls of the website
# this endpoint will crawl all the webpages of the site and train it on the collection
@router.post("/sitemapmethod/")
def train_sitemap(sitemap: str ,index_num: int): 
    sitemap_loader = SitemapLoader(web_path=sitemap)
    sitemap_loader.requests_kwargs = {"verify": False}
    docs = sitemap_loader.load()
    for i in docs:
        add_doc([i.page_content], index_num)
    return "Done"

# endpoint to fetch text from pdf and add to collection
@router.post("/pdfmethod/")
def train_pdf(pdfurl: str ,index_num: int): 
    loader = PyPDFLoader(pdfurl)
    pages = loader.load_and_split()
    for i in pages:
        add_doc([i.page_content], index_num)
    return "Done"
