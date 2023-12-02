from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings
from flask import Flask, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from llama_index import SimpleDirectoryReader
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hashlib
import openai
import pinecone
import mimetypes
from langchain.llms import OpenAI
from open_ai.shopping.query import getResponse
from utils.process import getCSV, getJSON, getWebsite
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader, UnstructuredPDFLoader, CSVLoader
openai.api_key = "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"

# Initialize OpenAI embeddings model
embeddings = OpenAIEmbeddings(
    openai_api_key="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")

openai.api_key = "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"


PINECONE_API_KEY = '2f1f9a16-8e97-4485-b643-bbcd3618570a'
PINECONE_ENVIRONMENT = 'us-west1-gcp-free'

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)


index = pinecone.Index('wing-sandbox')
index.delete(delete_all=True)

openai_api_key = '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'
tokenizer = tiktoken.get_encoding('cl100k_base')

BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYW1lIjoiQXNod2luIENoaXJ1bWFtaWxsYSJ9.keW___VBKcQY6uyxkxOH_uXZ1Jo74171cVa8SozxrKc"


datas = []
docsearchers = []
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=0)


UPSERT_BATCH_SIZE = 100

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000/"])

# unblock CORS


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin',
                         'http://localhost:3000/')
    response.headers.add('Access-Control-Allow-Headers',
                         'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods',
                         'GET,PUT,POST,DELETE,OPTIONS,PATCH')
    return response


def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def process_csv(text):
    chunks = text_splitter.split_text(text)
    documents = []

    for i, chunk in enumerate(chunks):
        documents.append({
            'id': str(hash(chunk)),
            'text': chunk
        })

    return documents


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
)


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join('data', filename)
    file.save(file_path)

    file_type, _ = mimetypes.guess_type(file_path)
    extension = file_type.split('/')[-1]

    if extension == 'pdf':
        loader = PyPDFLoader(file_path)
        data = loader.load()
        datas.append(data)

    elif extension == 'csv':
        loader = CSVLoader(file_path)
        data = loader.load()
        datas.append(data)
    texts = text_splitter.split_documents(data)
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name="wing-sandbox")
    docsearchers.append(docsearch)
    
    return "file successfully uploaded"


@app.route('/query', methods=['POST'])
def query_chat():
    data = request.json
    query = data.get('query_string')
    user_profile = "My name is Arth Bohra and I live in the extreme cold."
    return getResponse(query, user_profile, docsearchers[0])


def load_document(filename):
    loader = PyPDFLoader(filename)
    docs = loader.load()

    return docs


def process_documents(docs):
    documents = []
    for doc in docs:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):

            documents.append({
                'id': str(hash(chunk)),
                'text': chunk
            })
    return documents


def get_embeddings(texts):
    response = openai.Embedding.create(
        input=texts, model="text-embedding-ada-002")
    data = response["data"]
    return [result["embedding"] for result in data]


def remove_files_from_data():
    for file in os.listdir('data'):
        os.remove(os.path.join('data', file))
