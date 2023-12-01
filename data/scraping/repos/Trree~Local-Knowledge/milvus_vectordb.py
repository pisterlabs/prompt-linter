from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from tenacity import retry, wait_random_exponential, stop_after_attempt

from src.codevecdb.config.Config import Config
from src.codevecdb.llmcache import cache_initialize

cfg = Config()

if cfg.milvus_secure == "True" or cfg.milvus_secure == "true":
    secure = True
else:
    secure = False

connection_args = {
    "host": cfg.milvus_host,
    "port": cfg.milvus_port,
    "secure": secure,
    "user": cfg.milvus_user,
    "password": cfg.milvus_password
}

cache_initialize()


@retry(reraise=True, wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def insert_doc_db(docs_list):

    collection_name = cfg.milvus_collection_name
    embeddings = OpenAIEmbeddings(model="ada")
    vector_store = Milvus(collection_name=collection_name, embedding_function=embeddings,
                          connection_args=connection_args)\
        .from_documents(
        docs_list,
        embedding=embeddings,
        collection_name=collection_name,
        connection_args=connection_args
    )
    return vector_store


@retry(reraise=True, wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def search_db(question):
    collection_name = cfg.milvus_collection_name
    embeddings = OpenAIEmbeddings(model="ada")
    docs = Milvus(collection_name=collection_name, embedding_function=embeddings,
                  connection_args=connection_args)\
        .similarity_search(query=question, collection_nam=collection_name)
    return docs
