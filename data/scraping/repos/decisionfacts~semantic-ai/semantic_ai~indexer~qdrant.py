from typing import Optional, Any
from aiopath import AsyncPath

from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient

from langchain.embeddings.base import Embeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from semantic_ai.indexer.base import BaseIndexer
from semantic_ai.utils import file_process, check_isfile, iter_to_aiter


class QdrantIndexer(BaseIndexer):
    """Create qdrant indexer and create client object

        Args:
            location:
                If `:memory:` - use in-memory Qdrant instance.
                If `str` - use it as a `url` parameter.
                If `None` - use default values for `host` and `port`.
            url: either host or str of "Optional[scheme], host, Optional[port], Optional[prefix]".
                Default: `None`
            port: Port of the REST API interface. Default: 6333
            grpc_port: Port of the gRPC interface. Default: 6334
            prefer_grpc: If `true` - use gPRC interface whenever possible in custom methods.
            https: If `true` - use HTTPS(SSL) protocol. Default: `None`
            api_key: API key for authentication in Qdrant Cloud. Default: `None`
            prefix:
                If not `None` - add `prefix` to the REST URL path.
                Example: `service/v1` will result in `http://localhost:6333/service/v1/{qdrant-endpoint}` for REST API.
                Default: `None`
            timeout:
                Timeout for REST and gRPC API requests.
                Default: 5.0 seconds for REST and unlimited for gRPC
            host: Host name of Qdrant service. If url and host are None, set to 'localhost'.
                Default: `None`
            path: Persistence path for QdrantLocal. Default: `None`
            **kwargs: Additional arguments passed directly into REST client initialization

        Example:
        . code-block:: python

            from semantic_ai.indexer import QdrantIndexer

            collection_name = "MyCollection"
            qdrant = QdrantIndexer(url, collection_name, embeddings)
        """

    CONTENT_KEY = "page_content"
    METADATA_KEY = "metadata"
    VECTOR_NAME = None

    def __init__(self,
                 index_name: str,
                 embedding: Optional[Embeddings] = HuggingFaceEmbeddings(),
                 content_payload_key: str = CONTENT_KEY,
                 metadata_payload_key: str = METADATA_KEY,
                 distance_strategy: str = "COSINE",
                 vector_name: Optional[str] = VECTOR_NAME,
                 location: Optional[str] = None,
                 url: Optional[str] = None,
                 port: Optional[int] = 6333,
                 grpc_port: int = 6334,
                 prefer_grpc: bool = False,
                 https: Optional[bool] = None,
                 api_key: Optional[str] = None,
                 prefix: Optional[str] = None,
                 timeout: Optional[float] = None,
                 host: Optional[str] = None,
                 path: Optional[str] = None,
                 **kwargs: Any,
                 ):
        self.location = location
        self.url = url
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.https = https
        self.api_key = api_key
        self.prefix = prefix
        self.timeout = timeout
        self.host = host
        self.path = path
        self.collection_name = index_name
        self.embeddings = embedding
        self.content_payload_key = content_payload_key
        self.metadata_payload_key = metadata_payload_key
        self.distance_strategy = distance_strategy
        self.vector_name = vector_name

        self.client = QdrantClient(
            location=self.location,
            url=self.url,
            port=self.port,
            grpc_port=self.grpc_port,
            prefer_grpc=self.prefer_grpc,
            https=self.https,
            api_key=self.api_key,
            prefix=self.prefix,
            timeout=self.timeout,
            host=self.host,
            path=self.path,
            **kwargs
        )

    async def create(self) -> Qdrant:
        return Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings,
            content_payload_key=self.content_payload_key,
            metadata_payload_key=self.metadata_payload_key,
            distance_strategy=self.distance_strategy,
            vector_name=self.vector_name
        )

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

    async def index(self, extracted_json_dir: str):
        if extracted_json_dir:
            documents = await self.from_documents(extracted_json_dir)
            if await check_isfile(extracted_json_dir):
                try:
                    await Qdrant.afrom_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        url=self.url,
                        api_key=self.api_key,
                        collection_name=self.collection_name
                    )
                except Exception as ex:
                    print(f"{ex}")
            else:
                try:
                    async for docs in iter_to_aiter(documents):
                        await Qdrant.afrom_documents(
                            documents=docs,
                            embedding=self.embeddings,
                            url=self.url,
                            api_key=self.api_key,
                            collection_name=self.collection_name
                        )
                except Exception as ex:
                    print(f"{ex}")
        else:
            raise ValueError(f"Please give valid file or directory path.")
