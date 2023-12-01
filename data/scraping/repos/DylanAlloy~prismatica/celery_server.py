from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pymongo import MongoClient
from bson import ObjectId
from pydantic import BaseModel
import hashlib
import os
from transformers import pipeline, AutoTokenizer
from typing import Optional
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader

app = FastAPI()
app.mount("/public", StaticFiles(directory="public"), name="public")

import configparser

config = configparser.ConfigParser()
config.read('config.ini')

mongo_host = config.get('mongo', 'host')
qdrant_host = config.get('qdrant', 'host')

class Document(BaseModel):
    filename: str
    question: str

class MongoDB:
    def __init__(self, uri: str, registry: Optional[str] = None):
        self._client = MongoClient(host=uri)
        self._registry = registry

    def __getattr__(self, name):
        return getattr(self._client, name)

    @property
    def type_registry(self):
        return self._registry

    def __getitem__(self, key):
        db = self._client[key]
        return MongoCollection(db)


class MongoCollection:
    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        return getattr(self._db, name)

    def __getitem__(self, key):
        return self._db[key]


mongo_client = MongoDB(mongo_host, registry='utf-8')
ml_tasks = mongo_client['tasks']['ml_tasks']
uploads_collection = mongo_client['user_uploads']['uploads']


@app.get("/")
async def read_index_html():
    with open("index.html", "r") as index_file:
        content = index_file.read()
    return HTMLResponse(content=content)

@app.get("/uploads")
async def get_uploads():
    # retrieve all documents from the uploads collection
    all_uploads = uploads_collection.find()
    # convert the documents to a list and remove the '_id' field from each document
    all_uploads = [upload for upload in all_uploads]
    for upload in all_uploads:
        upload.pop('_id', None)
    # return the list of documents
    return all_uploads

@app.get("/tasks")
async def get_tasks():
    # retrieve all documents from the uploads collection
    all_tasks = ml_tasks.find()
    # convert the documents to a list and remove the '_id' field from each document
    all_tasks = [task for task in all_tasks]
    # return the list of documents
    return all_tasks

def generate_text_task(text: str):
    generated_text = generator(text, max_length=100)
    task_id = str(ObjectId())
    task_result = generated_text[0]['generated_text']
    print(task_result)
    task = {
        '_id': task_id,
        'status': 'done',
        'result': task_result
    }
    ml_tasks.insert_one(task)
    return task_id

@app.post("/generate_text")
async def generate_text(text: dict, background_tasks: BackgroundTasks):
    task_id = str(ObjectId())
    task = {
        '_id': task_id,
        'status': 'processing',
        'result': None
    }
    ml_tasks.insert_one(task)
    background_tasks.add_task(generate_text_task, text['text'])
    return {"task_id": task_id}

def is_valid_filetype(filename):
    allowed_extensions = ['.pdf', '.docx', '.txt', '.log']
    file_extension = os.path.splitext(filename)[1]
    return file_extension in allowed_extensions

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    if not is_valid_filetype(file.filename):
        raise HTTPException(
            status_code=400, detail="Invalid filetype. Allowed filetypes are: .pdf, .docx, .txt")

    file_extension = os.path.splitext(file.filename)[1]
    file_md5 = hashlib.md5(file.file.read()).hexdigest()
    file.file.seek(0)
    file_path = f"uploads/{os.path.splitext(file.filename)[0]+file_md5[0:8]}{file_extension}"

    # Check if file with same md5 exists in MongoDB
    existing_file = uploads_collection.find_one({'md5': file_md5})
    if existing_file:
        raise HTTPException(
            status_code=400, detail="File with same MD5 already exists in the database")

    with open(file_path, "wb") as f:
        f.write(file.file.read())
    file_info = {
        'filename': os.path.splitext(file.filename)[0]+file_md5[0:8]+file_extension,
        'md5': file_md5,
        'path': file_path
    }
    uploads_collection.insert_one(file_info)
    return {"filename": file_info["filename"], "md5": file_md5, "path": file_path}

@app.get("/task_status/{task_id}")
async def task_status(task_id: str):
    task = ml_tasks.find_one({'_id': task_id})
    if task is None:
        return {"status": "not found"}
    else:
        return task

async def find_similar_documents_task(document: Document, task: str):
    print(document)
    embedding_model = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    _question = document.question
    _limit = 10
    client = QdrantClient(url=qdrant_host+":6333")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=250,
        chunk_overlap=0,
        length_function=len,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model
    )
    # Load document from file
    if document.filename.split(".")[1] == 'pdf':
        loader = PyPDFLoader(f'uploads/{document.filename}')
    else:
        loader = Docx2txtLoader(f'uploads/{document.filename}')
    data = loader.load_and_split()

    # Split document into pages
    docs = []
    for each in data:
        _page_content = text_splitter.create_documents([each.page_content])
        for page in _page_content:
            doc = page.page_content.replace("\n", " ").replace("\t", " ")
            docs.append(doc)
            # _id=[str(each.metadata["page"])]
    Qdrant.from_texts(
        texts=docs, embedding=embeddings, host=qdrant_host, collection_name="custom_llm"
    )
    qdrant = Qdrant(
        client=client, collection_name="custom_llm",
        embeddings=embeddings.embed_query
    )

    query = _question
    found_docs = qdrant.similarity_search_with_score(query, k=_limit)

    def concat_page_content(docs):
        for doc, _ in docs:
            yield doc.page_content.replace("\n", "")

    page_content_generator = concat_page_content(found_docs)

     # Save search results to database
    result = list(page_content_generator)
    ml_tasks.update_one(
        {'_id': task},
        {'$set': {'status': 'done', 'result': result}}
    )

@app.post("/find_similar_documents")
async def find_similar_documents(document: Document, background_tasks: BackgroundTasks):
    print(document)
    task_id = str(ObjectId())
    task = {
        '_id': task_id,
        'status': 'processing',
        'result': None
    }
    ml_tasks.insert_one(task)
    background_tasks.add_task(find_similar_documents_task, document, task_id)
    return {"task_id": task_id}

async def save_search_results(found_docs):
    # TODO: Save search results to database
    pass
