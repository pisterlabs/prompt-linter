# =========================
#  Module: Vector DB Build
# =========================
import box
import yaml
import os
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredMarkdownLoader

# Import config vars
with open('code_qa/config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


# Build vector database
def run_db_build():
    loader = DirectoryLoader(cfg.DATA_PATH,
                             glob='**/*.cs.summary.txt',
                             loader_cls=UnstructuredMarkdownLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                   chunk_overlap=cfg.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)

    api_key = os.environ['OPENAI_API_KEY_ALTINN3_DEV']
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(cfg.DB_FAISS_PATH)

if __name__ == "__main__":
    run_db_build()
