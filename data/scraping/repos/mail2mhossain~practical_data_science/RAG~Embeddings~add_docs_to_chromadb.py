import os
import glob
import textwrap

from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader  # for textfiles
from langchain.text_splitter import CharacterTextSplitter  # text splitter
from langchain.embeddings import HuggingFaceEmbeddings  # for using HugginFace models
from langchain.document_loaders import UnstructuredPDFLoader  # load pdf


persist_directory = "../ChromaDB"

# Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)

# Embeddings
embeddings = HuggingFaceEmbeddings()

# PDF loader
pdf_folder_path = "../New_Documents/"
os.listdir(pdf_folder_path)
pdf_files = glob.glob(os.path.join(pdf_folder_path, f"*{'.pdf'}"))

print(pdf_files)

docs = [
    text_splitter.split_documents(UnstructuredPDFLoader(fn).load()) for fn in pdf_files
]

print(len(docs))

# Initialize PeristedChromaDB
chroma = Chroma(
    collection_name="corporate_db",
    embedding_function=embeddings,
    persist_directory=persist_directory,
)
chroma.add_documents(docs)
