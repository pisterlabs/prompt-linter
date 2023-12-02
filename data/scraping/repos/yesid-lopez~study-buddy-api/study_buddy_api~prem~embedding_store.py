import os

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis

from study_buddy_api.utils.logger import get_logger

os.environ["OPENAI_API_KEY"] = "random-string"
log = get_logger(__file__)


class EmbeddingStore:
    def __init__(self, index_name=None):
        self.redis_url = os.getenv("PREM_REDIS_URI")
        self.embeddings = OpenAIEmbeddings(openai_api_base=f"{os.getenv('PREM_EMBEDDINGS_BASE')}")
        self.vector_store = None
        if index_name:
            self.vector_store = Redis(
                redis_url=self.redis_url,
                index_name=index_name,
                embedding_function=self.embeddings.embed_query
            )
        self.index_name = index_name

    def save_content(self, docs: list[Document], file_name: str):
        Redis.drop_index(file_name, True, redis_url=self.redis_url)
        vector_store = Redis.from_documents(docs, self.embeddings, redis_url=self.redis_url, index_name=f"{file_name}")
        return vector_store.index_name

    def query(self, question, n_results):
        docs = self.vector_store.similarity_search(question, n_results)
        return docs
