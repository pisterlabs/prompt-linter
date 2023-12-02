import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from vector.vector_store import get_vector_store

db = get_vector_store(dataset_name = "met_office_api_docs")

document_loader = PyPDFLoader(file_path="data/met_office/datapoint_api_reference.pdf")
document = document_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
docs = text_splitter.split_documents(document)
print("Deleting documents in vector store")
db.delete(delete_all=True)
print("Storing document in vector store")
db.add_documents(docs)
