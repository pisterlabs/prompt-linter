from pathlib import Path
import os
from app.core.logger import get_logger
logger = get_logger('langchain')
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader, UnstructuredExcelLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
import pinecone

load_dotenv()
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'),
              environment="us-east1-gcp")


def load_and_split_txt(file_path: str, namespace: str, file_name: str, doc_id: str):
    loader = TextLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    for doc in split_docs:
        doc.metadata['namespace'] = namespace
        doc.metadata['file_name'] = file_name
        doc.metadata['doc_id'] = doc_id

    return split_docs


def load_and_split_pdf(file_path: str, namespace: str, file_name: str, doc_id: str):
    # TODO: Explore what the best PDF loader seems to be. I believe it is "unstructured", but the dependencies are crazy
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    for doc in split_docs:
        doc.metadata['namespace'] = namespace
        doc.metadata['file_name'] = file_name
        doc.metadata['doc_id'] = doc_id
    return split_docs


def load_and_split_doc(file_path: str, namespace: str, file_name: str, doc_id: str):
    # TODO: Explore what the best doc loader seems to be. I believe it is "unstructured", but the dependencies are crazy
    loader = Docx2txtLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    for doc in split_docs:
        doc.metadata['namespace'] = namespace
        doc.metadata['file_name'] = file_name
        doc.metadata['doc_id'] = doc_id
    return split_docs

def load_and_split_csv(file_path: str, namespace: str, file_name: str, doc_id: str):
    # TODO: Explore what the best doc loader seems to be. I believe it is "unstructured", but the dependencies are crazy
    loader = CSVLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata['namespace'] = namespace
        doc.metadata['file_name'] = file_name
        doc.metadata['doc_id'] = doc_id
    return docs

def load_and_split_excel(file_path: str, namespace: str, file_name: str, doc_id: str):
    # TODO: Explore what the best doc loader seems to be. I believe it is "unstructured", but the dependencies are crazy
    loader = UnstructuredExcelLoader(file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata['namespace'] = namespace
        doc.metadata['file_name'] = file_name
        doc.metadata['doc_id'] = doc_id
    return docs

def create_pinecone_index(documents: list, namespace: str):
    '''namespace is monday_account_id + _ + chatbotname'''
    embeddings = OpenAIEmbeddings()
    with get_openai_callback() as cb:
        vector_db = Pinecone.from_documents(documents=documents, embedding=embeddings, index_name='obo-internal-slackbot', namespace=namespace)
        tokens_used = cb.total_tokens
        logger.debug(f'CREATE_INDEX: Tokens Used: {tokens_used}')
    return vector_db