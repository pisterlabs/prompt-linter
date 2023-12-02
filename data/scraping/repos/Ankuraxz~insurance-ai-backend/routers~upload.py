import os


from langchain.vectorstores import Milvus
from fastapi import APIRouter, Header, File, UploadFile
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import HTTPException
from typing import Annotated, Union
import logging
import uuid
import shutil

Logger = logging.getLogger(__name__)
Logger.setLevel(logging.INFO)

router = APIRouter()
upload_dir = os.path.join(os.getcwd(), "uploads")


def text_splitter(docs, chunk_size=1000, chunk_overlap=0):
    """
    Splits the documents into chunks
    :param docs:
    :param chunk_size:
    :param chunk_overlap:
    :return:
    """
    text_splitter_func = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter_func.split_documents(docs)
    return docs


@router.post("/upload", tags=["upload"])
async def create_upload_file(email_id: Annotated[Union[str, None], Header()], file: UploadFile = File(...)):
    """
    Uploads a file to the server
    :param email_id:
    :param file:
    :return:
    """
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    file_location = os.path.join(upload_dir, (str(uuid.uuid4()) + file.filename))
    try:
        if file.filename.endswith('.txt'):
            with open(file_location, "wb") as file_object:
                shutil.copyfileobj(file.file, file_object)
            loader = TextLoader(file_location)
            documents = loader.load()

        elif file.filename.endswith('.pdf'):
            with open(file_location, "wb") as file_object:
                shutil.copyfileobj(file.file, file_object)
            loader = PyPDFLoader(file_location)
            documents = loader.load_and_split()
        else:
            return {"message": "File not supported"}
    except Exception as e:
        Logger.error(f"An Exception Occurred while loading file --> {e}")
        raise HTTPException(status_code=404, detail=f"Error in loading file --> {e}")

    try:
        docs = text_splitter(documents)
        embeddings = OpenAIEmbeddings()
        Milvus.from_documents(
            docs,
            embeddings,
            collection_name=email_id.split("@")[0],
            connection_args={
                "host":os.environ.get('VM_HOST'),
                "port": "19530",
            },
        )
        Logger.info(f"File uploaded successfully")
        os.remove(file_location)
        return {"message": "File uploaded successfully"}
    except Exception as e:
        Logger.error(f"An Exception Occurred while uploading file --> {e}")
        raise HTTPException(status_code=404, detail=f"Error in uploading file --> {e}")
