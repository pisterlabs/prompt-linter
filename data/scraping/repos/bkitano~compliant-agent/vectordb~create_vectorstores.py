from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from pathlib import Path
import os

docs = os.listdir(Path(__file__).parent.parent / "docs")
# for demo:
# docs = ["us-fcpa.txt", "ice-cream.txt"]
doc_paths = [Path(__file__).parent.parent / f"docs/{doc}" for doc in docs]

documents = [TextLoader(doc_path).load()[0] for doc_path in doc_paths]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
print(texts)

embeddings = HuggingFaceEmbeddings(model_name="gtr-t5-large")
docsearch = FAISS.from_documents(texts, embeddings)

docsearch.save_local("vectorstores/us-fcpa-vectorstore")
