from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb
import openai
import os
from configs import conf

embedding_model = "text-embedding-ada-002"


class KnowledgeDB:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(KnowledgeDB, cls).__new__(
                cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        # 初始化chromadb，chromadb官方地址：https://docs.trychroma.com/
        self.db = chromadb.PersistentClient(path=self.get_save_path()).get_or_create_collection(
            name='test',
            embedding_function=OpenAIEmbeddingFunction(
                api_key=conf.get("api_key"),
                model_name=embedding_model,
            ),
        )

    # 向 向量数据库 中插入数据
    def add(self, things=[]):
        openai.api_key = conf.get("api_key")

        # 调用OpenAI API 进行向量化
        embeddingThings = []
        for i in things:
            embeddingThings.append(i["text"])

        embeddingResults = openai.Embedding.create(
            input=embeddingThings, model=embedding_model)["data"]

        # 转换数据格式
        dbIDs = []
        dbEmbeddings = []
        dbDocs = []
        for i in range(len(things)):
            dbIDs.append(things[i]["id"])
            dbDocs.append(things[i]["text"])
            dbEmbeddings.append(embeddingResults[i]["embedding"])

        # 存储到数据库中
        self.db.add(
            ids=dbIDs,
            embeddings=dbEmbeddings,
            documents=dbDocs
        )

    def search(self, user_prompt: str) -> str:
        return self.db.query(query_texts=user_prompt, n_results=1, include=['distances', 'documents'])

    def get_save_path(self) -> str:
        return os.path.dirname(os.path.abspath(__file__)) + "/data/"
