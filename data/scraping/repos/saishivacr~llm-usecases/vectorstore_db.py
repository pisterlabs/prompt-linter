# =========================
#  Module: Vector DB Build
# =========================
import os
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from utils.load_Vars import *

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

knowledge_base_path = f"{project_root}/{DATA_PATH}"

# Build vector database
def run_db_build():
    try:
        loader = DirectoryLoader(knowledge_base_path,
                                glob='*.pdf',
                                loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                    chunk_overlap=CHUNK_OVERLAP)
        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                        model_kwargs={'device': 'cpu'})

        vectorstore = FAISS.from_documents(docs, embeddings)
        #vectorstore.save_local(DB_FAISS_PATH)
        return vectorstore
    except Exception as e:
        error_msg = f"An error occurred while reading files: {e}"
        return None
if __name__ == "__main__":
    run_db_build()
