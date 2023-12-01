# =========================
#  Module: Vector DB Build
# =========================
import box
import yaml
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from src.logger import logging
import os

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


# Build vector database
def run_db_build(chat_id):
    # load pdf files
    # loader = DirectoryLoader(cfg.DATA_PATH,
    #                          glob='*.pdf',
    #                          loader_cls=PyPDFLoader)
    # documents = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
    #                                                chunk_overlap=cfg.CHUNK_OVERLAP)
    # texts = text_splitter.split_documents(documents)

    # NEED TO CHECK
    # loader = DirectoryLoader('../', glob="**/*.md", loader_cls=TextLoader)

    if not os.path.exists(f'{cfg.DB_FAISS_PATH}/{chat_id}'):
        os.makedirs(f'{cfg.DB_FAISS_PATH}/{chat_id}')
        logging.info(f"Folder '{f'{cfg.DB_FAISS_PATH}/{chat_id}'}' created.")

        file_list = os.listdir(f'{cfg.DATA_PATH}/{chat_id}')
        txt_files = [file for file in file_list if file.endswith(".txt")]

        files = []
        for file in txt_files:
            with open(f'{cfg.DATA_PATH}/{chat_id}/{file}') as f:
                state_of_the_union = f.read()
                files.append(state_of_the_union)

        logging.info("Load all files to create Vector Database")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                    chunk_overlap=cfg.CHUNK_OVERLAP)

        texts = text_splitter.create_documents(files)

        logging.info("Create chunks using text splitter")
        
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs={'device': 'cpu'})
        
        logging.info("Initialize embedding model - Sentence BERT")

        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(f'{cfg.DB_FAISS_PATH}/{chat_id}')

        logging.info("Save files into Vector Database")

    else:
        logging.info(f"Folder '{f'{cfg.DB_FAISS_PATH}/{chat_id}'}' already exists.")

    # vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=".")
    # vectorstore.save_local(cfg.DB_CHROMA_PATH)
    # vectorstore.persist()

if __name__ == "__main__":
    run_db_build()
