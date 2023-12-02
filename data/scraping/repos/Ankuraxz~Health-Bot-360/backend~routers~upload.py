import logging
import os
import shutil
import uuid
from typing import Annotated, Union
from pymongo import MongoClient
import certifi

from fastapi import APIRouter, Header, File, UploadFile
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import mongodb_atlas
from langchain.vectorstores import MongoDBAtlasVectorSearch
from fastapi import HTTPException

Logger = logging.getLogger(__name__)
Logger.setLevel(logging.INFO)

router = APIRouter()

upload_dir = "./uploads"
client = MongoClient(os.environ.get('MONGO_URI'), ssl=True, tlsCAFile=certifi.where())
db = client["KNN"]
collection = db["embeddings"]




# Helpers
def text_splitter(docs, chunk_size=1000, chunk_overlap=20):
    text_splitter_func = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter_func.split_documents(docs)
    return docs

@router.post("/uploadfile/", tags=["upload"])
async def create_upload_file(email_id: Annotated[Union[str, None], Header()], file: UploadFile = File(...)):
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_location = os.path.join(upload_dir, (str(uuid.uuid4()) + file.filename))
    try:
        if file.filename.endswith('.txt'):
            with open(file_location, "wb") as file_object:
                shutil.copyfileobj(file.file, file_object)
            loader = TextLoader(file_location)

        elif file.filename.endswith('.pdf'):
            with open(file_location, "wb") as file_object:
                shutil.copyfileobj(file.file, file_object)
            loader = PyPDFLoader(file_location)
        else:
            return {"message": "File not supported"}
    except Exception as e:
        Logger.error(f"An Exception Occurred while loading file --> {e}")
        raise HTTPException(status_code=404, detail=f"Error in loading file --> {e}")

    documents = loader.load()
    docsf = text_splitter(documents)
    try:
        MongoDBAtlasVectorSearch.from_documents(
            documents=docsf,
            embedding=OpenAIEmbeddings(disallowed_special=()),
            collection=db["embeddings"],
            index_name="vector-search",
        )
        Logger.info(f"Data written to mongo db")
        return {"status": "success"}
    except Exception as e:
        Logger.error(f"An Exception Occurred while writing to mongo db --> {e}")
        raise HTTPException(status_code=404, detail=f"Error in writing to mongo db --> {e}")
