from __future__ import annotations

import client.swagger_client as swagger_client
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
from langchain.utils import get_from_dict_or_env
from langchain.vectorstores.base import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from client.swagger_client.rest import ApiException
import uuid
import datetime
from pprint import pprint

def _json_serializable(value: Any) -> Any:
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    return value

class FaVeClient(VectorStore):
    def __init__(
            self,
            collection: str,
            text_key: str,
            url: str = "http://localhost:1234",
            embedding: Optional[Embeddings] = None,
            client: Optional[Any] = None
    ) -> None:
        self._url = url
        self._collection = collection
        self._text_key = text_key
        self._embedding = embedding
        configuration = swagger_client.Configuration()
        configuration.host = url+"/v1"

        if client is not None:
            self._client = client
        else:
            self._client = swagger_client.DefaultApi(swagger_client.ApiClient(configuration))

        @property
        def embeddings(self) -> Optional[Embeddings]:
            return self._embedding
        
        # TODO check if collection exists

        # if not create collection
        colection = swagger_client.Collection()
        colection.name = collection
        colection.indexes = []
        try:
            api_response = self._client.fave_create_collection(colection)
        except ApiException as e:
            raise Exception("%s\n" % e)

    def add_texts(
        self, texts: Iterable[str], metadatas: Optional[List[dict]] = None, **kwargs: Any,
    ) -> List[str]:
        ids = []
        embeddings: Optional[List[List[float]]] = None
        if self._embedding:
            if not isinstance(texts, list):
                texts = list(texts)
            embeddings = self._embedding.embed_documents(texts)

        properties_to_vectorize = []
        if embeddings is None:
            if "props_to_index" in kwargs:
                properties_to_vectorize = kwargs["props_to_index"]

        rqst = swagger_client.AddDocumentsRequest()
        rqst.name = self._collection
        documents = []

        for i, text in enumerate(texts):
                data_properties = {self._text_key: text}
                if metadatas is not None:
                    for key, val in metadatas[i].items():
                        data_properties[key] = _json_serializable(val)
                data_properties["vector"] = embeddings[i] if embeddings else None,
                _id = str(uuid.uuid4())
                document = swagger_client.Document()
                document.properties = data_properties
                document.id = _id
                documents.append(document)
                ids.append(_id)
        
        rqst.documents = documents
        rqst.properties_to_vectorize = properties_to_vectorize
        try:
            response = self._client.fave_add_documents(rqst)
            pprint(response)
        except ApiException as e:
            raise Exception("%s\n" % e)
        
        return ids  
    
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        rqst = swagger_client.NearestDocumentsRequest()
        rqst.name = self._collection
        rqst.text = query
        rqst.limit = k
        rqst.distance = 1.0

        try:
            resp = self._client.fave_get_nearest_documents(rqst)
            pprint(resp.name)
        except ApiException as e:
            raise Exception("%s\n" % e)
        docs = []
        for res in resp.documents:
            text = res.properties.pop(self._text_key)
            docs.append(Document(page_content=text, metadata=res.properties))

        pprint(docs)
        return docs 
    
    @classmethod
    def from_texts(
        cls: Type[FaVeClient],
        texts: List[str],
        collection: str = "text",
        text_key: str = "text",
        url: str = "http://localhost:1234",
        metadatas: Optional[List[dict]] = None,
        embedding: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> FaVeClient:
        properties_to_vectorize = []
        embeddings: Optional[List[List[float]]] = None
        if embedding is not None:
            embeddings = embedding.embed_documents(texts) if embedding else None
        else:
            properties_to_vectorize = kwargs["props_to_index"]

        rqst = swagger_client.AddDocumentsRequest()
        rqst.name = collection
        documents = []

        for i, text in enumerate(texts):
            data_properties = {text_key: text}
            if metadatas is not None:
                for key, val in metadatas[i].items():
                    data_properties[key] = _json_serializable(val)
                
            if embeddings is not None:
                data_properties["vector"] = embeddings[i]

            document = swagger_client.Document()
            document.properties = data_properties
            document.id = str(uuid.uuid4())
            documents.append(document)

        rqst.documents = documents
        rqst.properties_to_vectorize = properties_to_vectorize

        configuration = swagger_client.Configuration()
        configuration.host = url+"/v1"
    
        client = swagger_client.DefaultApi(swagger_client.ApiClient(configuration))
        
        # TODO check if collection exists

        # if not create collection
        colection = swagger_client.Collection()
        colection.name = collection
        colection.indexes = []
        try:
            client.fave_create_collection(colection)
        except ApiException as e:
            raise Exception("%s\n" % e)
        
        try:
            client.fave_add_documents(rqst)
        except ApiException as e:
            raise Exception("%s\n" % e)
        
        return cls(collection, text_key, url, embedding, client)
