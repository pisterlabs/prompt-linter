from aiopath import AsyncPath

from typing import (
    Optional
)

from langchain.embeddings.base import Embeddings
from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from semantic_ai.indexer.base import BaseIndexer
from semantic_ai.utils import file_process, check_isfile, iter_to_aiter

from elasticsearch import Elasticsearch


class ElasticsearchIndexer(BaseIndexer):

    def __init__(
            self,
            *,
            url: str,
            es_user: str | None = None,
            es_password: str | None = None,
            index_name: str,
            embedding: Optional[Embeddings] = HuggingFaceEmbeddings(),
            verify_certs: bool = True,
            es_api_key: Optional[str] = None
    ):
        super().__init__()
        self.url = url
        self.es_user = es_user
        self.es_password = es_password
        self.index_name = index_name
        self.embeddings = embedding
        self.verify_certs = verify_certs
        self.es_api_key = es_api_key

        self.es_connection = Elasticsearch(self.url,
                                           basic_auth=(self.es_user, self.es_password),
                                           verify_certs=self.verify_certs
                                           )

    async def create(self) -> ElasticsearchStore:
        if not self.verify_certs:
            obj = ElasticsearchStore(
                embedding=self.embeddings,
                index_name=f"{self.index_name}",
                es_connection=self.es_connection
            )
        else:
            obj = ElasticsearchStore(
                embedding=self.embeddings,
                es_url=self.url,
                es_user=self.es_user,
                es_password=self.es_password,
                index_name=f"{self.index_name}",
                es_api_key=self.es_api_key
            )
        return obj

    @staticmethod
    async def from_documents(extracted_json_dir):
        if extracted_json_dir:
            datas = []
            dir_path = AsyncPath(extracted_json_dir)
            if await dir_path.is_file():
                file_path = str(dir_path)
                file_ext = dir_path.suffix.lower()
                data = await file_process(file_ext=file_ext, file_path=file_path)
                return data
            elif await dir_path.is_dir():
                async for path in dir_path.iterdir():
                    if await path.is_file():
                        file_path = str(path)
                        file_ext = path.suffix.lower()
                        _data = await file_process(file_ext=file_ext, file_path=file_path)
                        datas.append(_data)
                    else:
                        pass
                return datas
        else:
            raise ValueError(f"Please give valid file or directory path.")

    async def index(self, extracted_json_dir_or_file: str):
        if extracted_json_dir_or_file:
            documents = await self.from_documents(extracted_json_dir_or_file)
            if await check_isfile(extracted_json_dir_or_file):
                try:
                    await ElasticsearchStore.afrom_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        es_url=self.url,
                        es_user=self.es_user,
                        es_password=self.es_password,
                        index_name=self.index_name
                    )
                except Exception as ex:
                    print(f"{ex}")
            else:
                try:
                    async for docs in iter_to_aiter(documents):
                        await ElasticsearchStore.afrom_documents(
                            documents=docs,
                            embedding=self.embeddings,
                            es_url=self.url,
                            es_user=self.es_user,
                            es_password=self.es_password,
                            index_name=self.index_name
                        )
                except Exception as ex:
                    print(f"{ex}")
        else:
            raise ValueError(f"Please give valid file or directory path.")
