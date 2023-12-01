import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from softtek_llm.chatbot import Chatbot
from softtek_llm.models import OpenAI
from softtek_llm.cache import Cache
from softtek_llm.vectorStores import PineconeVectorStore
from softtek_llm.embeddings import OpenAIEmbeddings
from softtek_llm.schemas import Filter
from softtek_llm.memory import Memory
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()

class Vector:
    def __init__(self, id, embeddings, metadata=None):
        self.id = id
        self.embeddings = embeddings
        self.metadata = metadata

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY not found in .env file")

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
if OPENAI_API_BASE is None:
    raise ValueError("OPENAI_API_BASE not found in .env file")

OPENAI_EMBEDDINGS_MODEL_NAME = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME")
if OPENAI_EMBEDDINGS_MODEL_NAME is None:
    raise ValueError("OPENAI_EMBEDDINGS_MODEL_NAME not found in .env file")

OPENAI_CHAT_MODEL_NAME = os.getenv("OPENAI_CHAT_MODEL_NAME")
if OPENAI_CHAT_MODEL_NAME is None:
    raise ValueError("OPENAI_CHAT_MODEL_NAME not found in .env file")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if PINECONE_API_KEY is None:
    raise ValueError("PINECONE_API_KEY not found in .env file")

PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
if PINECONE_ENVIRONMENT is None:
    raise ValueError("PINECONE_ENVIRONMENT not found in .env file")

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
if PINECONE_INDEX_NAME is None:
    raise ValueError("PINECONE_INDEX_NAME not found in .env file")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def calculate_embeddings(text_chunk):
    embeddings = embeddings_model.embed(text_chunk)
    return embeddings

def convert_to_vectors(text_chunks):
    return [Vector(id=str(i), embeddings=calculate_embeddings(chunk)) for i, chunk in enumerate(text_chunks)]


def get_vectorstore(chunks,vector_store):
    vector_store.add(vectors=chunks)
    print("Se ha guardado correctamente")
    return vector_store


vector_store = PineconeVectorStore(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT,
    index_name=PINECONE_INDEX_NAME,
)
embeddings_model = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model_name=OPENAI_EMBEDDINGS_MODEL_NAME,
    api_type="azure",
    api_base=OPENAI_API_BASE,
)
cache = Cache(
    vector_store=vector_store,
    embeddings_model=embeddings_model,
)

memory = Memory()

model = OpenAI(
    api_key=OPENAI_API_KEY,
    model_name=OPENAI_CHAT_MODEL_NAME,
    api_type="azure",
    api_base=OPENAI_API_BASE,
    temperature=1,
    verbose=True,
)

filters = [
    Filter(
        type="DENY",
        case="Deny all bad words in english and spanish",
    )
]
chatbot = Chatbot(
    model=model,
    memory=memory,
    description="You are a virtual assintant",
    filters=filters,
    cache=cache,
    verbose=True,
)

raw_text = get_pdf_text(["cotizacion_piezotome_cube.pdf"])
memory.add_message("user",raw_text)
text_chunks = get_text_chunks(raw_text)
vector_chunks = convert_to_vectors(text_chunks)
vector_store.add(vector_chunks)
raw_text = get_pdf_text(["cotizacion_GBT.pdf"])
memory.add_message("user",raw_text)
text_chunks = get_text_chunks(raw_text)
vector_chunks = convert_to_vectors(text_chunks)
vector_store.add(vector_chunks)
raw_text = get_pdf_text(["cotizacion_piezotome_cube.pdf"])
memory.add_message("user",raw_text)
text_chunks = get_text_chunks(raw_text)
vector_chunks = convert_to_vectors(text_chunks)
vector_store.add(vector_chunks)

res = 0
while res == 0:
    respuesta = input("\n\n\nEscribe tu pregunta: ")
    response = chatbot.chat(
        respuesta,
        memory.add_message("user",respuesta),
    )
    print(response)
    res = input("\n\n\nQuieres salir?\n1) Si\n2) No\nEscribe tu respuesta: ")


