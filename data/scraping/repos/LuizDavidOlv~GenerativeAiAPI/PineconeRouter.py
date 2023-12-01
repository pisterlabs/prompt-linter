import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pinecone
from API.Models.PineconeCreateIndexModel import CreateIndexModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY'),
    environment=os.environ.get('PINECONE_ENV')
)

class Instructions(BaseModel):
    instruction: str

router = APIRouter(
    prefix="/pinecone",
    tags=["Pinecone"]
)

@router.get("/version/")
def version():
    try:
        return pinecone.info.version()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e}') 

@router.get("/list-indexes/")
def list_indexes():
    try:
        return pinecone.list_indexes()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e.body}') 

@router.post("/describe-index/")
def describe_index(index_name: str):
    try:
        return pinecone.describe_index(index_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e.body}')

@router.post("/create-index/")
def create_index(data: CreateIndexModel):
    try:
        return pinecone.create_index(data.index_name, dimension= data.dimension, metric = data.metric, pods= 1, pod_type='p1.x2')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e.body}') 
    
@router.post("/upsert-index/")
def upsert_index(index_name: str, text: str):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.create_documents([text])
        embeddings = OpenAIEmbeddings()
        return Pinecone.from_documents(chunks,embeddings,index_name=index_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e.body}')

@router.delete("/delete-index")
def delete_index(index_name: str):
    try:
        return pinecone.delete_index(index_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Error:  {e.body}')
    


