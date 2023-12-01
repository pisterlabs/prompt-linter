import base64
from logging import getLogger
import os
import traceback
from typing import List, Optional

import aiofiles
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from langchain.schema import Document as LDocument
from langchain.schema.embeddings import Embeddings
from langchain.document_loaders import PDFMinerLoader, TextLoader, CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma


logger = getLogger(__name__)


# API Schemas
class Document(BaseModel):
    page_content: str = Field(..., title="page_content", description="Content of the document", example="Eels and conger eels are both long, thin fish, but the difference is that eels are freshwater fish and conger eels are saltwater fish.")
    metadata: dict = Field(..., title="metadata", description="Metadata of the document as JSON", example={"source": "fish.pdf"})


class AddRequest(BaseModel):
    documents: List[Document] = Field(..., title="documents", description="List of documents to add")


class AddResponse(BaseModel):
    ids: List[str] = Field(..., title="ids", description="List of id of added documents", example=["001-aaa", "002-aaa", "003-aaa"])


class UploadRequest(BaseModel):
    b64content: str = Field(..., title="b64content", description="Base64 encoded content of the document", example="(b64 encoded data)")
    filename: str = Field(..., title="filename", description="Filename to be stored as 'source'", example="fish.pdf")
    document_type: str = Field(..., title="document_type", description="Extension starts with '.'", example=".pdf")
    loader_params: Optional[dict] = Field(None, title="loader_params", description="Parameters for loader", example={"jq_schema": ".records[].page_content"})


class UploadResponse(AddResponse):
    pass


class UpdateReqeust(BaseModel):
    ids: List[str] = Field(..., title="ids", description="List of id of documents to update", example=["001-aaa", "002-aaa"])
    documents: List[Document] = Field(..., title="documents", description="List of documents to update", examples=[[{"page_content": "Eels and conger eels are both long, thin fish, but the difference is that eels are freshwater fish and conger eels are saltwater fish.", "metadata": {"source": "fish.pdf"}}, {"page_content": "Red pandas are smaller than pandas, but when it comes to cuteness, there is no \"lesser\" about them.", "metadata": {"source": "fish.pdf"}}]])


class GetResponse(BaseModel):
    ids: List[str] = Field(..., title="ids", description="List of id of documents", example=["001-aaa", "002-aaa"])
    documents: List[Document] = Field(..., title="documents", description="List of documents", examples=[{"page_content": "Eels and conger eels are both long, thin fish, but the difference is that eels are freshwater fish and conger eels are saltwater fish."}, {"page_content": "Red pandas are smaller than pandas, but when it comes to cuteness, there is no \"lesser\" about them."}])


class SearchResponse(BaseModel):
    results: List[Document] = Field(..., title="results", description="Search results")


# API router
class LangChainVSSLiteServer:
    def __init__(self, apikey: str, persist_directory: str = "./vectorstore", chunk_size: int = 500, chunk_overlap: int = 0, embedding_function: Embeddings = None, server_args: dict = None):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function or OpenAIEmbeddings(openai_api_key=apikey)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.app = FastAPI(**(server_args or {"title": "VSSLite API", "version": "0.6.1"}))
        self.setup_handlers()

    def setup_handlers(self):
        app = self.app

        def get_vector_store(namespace: str = "default") -> Chroma:
            return Chroma(
                persist_directory=os.path.join(self.persist_directory, namespace),
                embedding_function=self.embedding_function
            )

        @app.get("/search/{namespace}", response_model=SearchResponse, tags=["Search"])
        async def search_document(q: str, count: int = 4, namespace: str = "default", score_threshold: float = 0.0):
            try:
                retriever = get_vector_store(namespace).as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": count, "score_threshold": score_threshold}
                )
                results = []
                for d in await retriever.aget_relevant_documents(query=q):
                    results.append(
                        Document(
                            page_content=d.page_content,
                            metadata=d.metadata
                        )
                    )

                sr = SearchResponse(results=results)

                return sr

            except Exception as ex:
                logger.error(f"Error at search_document: {ex}\n{traceback.format_exc()}")
                return JSONResponse({"error": "Internal server error"}, 500)

        def get_documents_chroma(ids: List[str] = None, namespace: str = "default") -> tuple[List[str], List[Document]]:
            # Chroma doesn't support async
            docs = get_vector_store(namespace).get(ids)
            return docs["ids"], [Document(
                page_content=docs["documents"][i], metadata=docs["metadatas"][i]
            ) for i in range(len(docs["documents"]))]

        @app.get("/document/{namespace}/all", response_model=GetResponse, tags=["Get"])
        async def get_all_documents(namespace: str = "default"):
            try:
                ids, documents = get_documents_chroma(namespace=namespace)
                return GetResponse(ids=ids, documents=documents)

            except Exception as ex:
                logger.error(f"Error at get_document: {ex}\n{traceback.format_exc()}")
                return JSONResponse({"error": "Internal server error"}, 500)

        @app.get("/document/{namespace}/{id}", response_model=GetResponse, tags=["Get"])
        async def get_document(id: str, namespace: str = "default"):
            ids, documents = get_documents_chroma([id], namespace)
            return GetResponse(ids=ids, documents=documents)

        @app.post("/document/{namespace}", response_model=AddResponse, tags=["Update"])
        async def add_documents(request: AddRequest, namespace: str = "default"):
            try:
                documents = [LDocument(
                    page_content=d.page_content,
                    metadata=d.metadata
                ) for d in request.documents]

                ids = await get_vector_store(namespace).aadd_documents(documents)

                return AddResponse(ids=ids)

            except Exception as ex:
                logger.error(f"Error at add_documents: {ex}\n{traceback.format_exc()}")
                return JSONResponse({"error": "Internal server error"}, 500)

        @app.patch("/document/{namespace}", tags=["Update"])
        async def update_documents(request: UpdateReqeust, namespace: str = "default"):
            try:
                get_vector_store(namespace).update_documents(
                    request.ids,
                    [LDocument(
                        page_content=d.page_content,
                        metadata=d.metadata
                    ) for d in request.documents]
                )
                return JSONResponse({})

            except Exception as ex:
                logger.error(f"Error at update_documents: {ex}\n{traceback.format_exc()}")
                return JSONResponse({"error": "Internal server error"}, 500)

        @app.post("/document/{namespace}/upload", response_model=UploadResponse, tags=["Update"])
        async def upload_document(request: UploadRequest, namespace: str = "default"):
            try:
                safe_filename = request.filename.replace("/", "_").replace("..", "_")
                binary_data = base64.b64decode(request.b64content)
                async with aiofiles.open(safe_filename, "wb") as file:
                    await file.write(binary_data)

            except Exception as ex:
                logger.error(f"Error at upload_document: {ex}\n{traceback.format_exc()}")
                return JSONResponse({"error": "Invalid content or filename"}, 400)

            try:
                loader_params = request.loader_params or {}

                if request.document_type.lower() == ".pdf":
                    loader = PDFMinerLoader(safe_filename, **loader_params)
                elif request.document_type.lower() == ".txt":
                    loader = TextLoader(safe_filename, **loader_params)
                elif request.document_type.lower() == ".csv":
                    loader = CSVLoader(safe_filename, **loader_params)
                elif request.document_type.lower() == ".json":
                    loader = JSONLoader(safe_filename, **loader_params)
                else:
                    return JSONResponse({"error": "Invalid document_type. We accept pdf or txt for now."}, 400)

                documents = loader.load()
                os.remove(safe_filename)

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                splited_documents = text_splitter.split_documents(documents)
                ids = await get_vector_store(namespace).aadd_documents(splited_documents)
                return AddResponse(ids=ids)

            except Exception as ex:
                logger.error(f"Error at upload_document: {ex}\n{traceback.format_exc()}")
                return JSONResponse({"error": "Internal server error"}, 500)

        @app.delete("/document/{namespace}/all", tags=["Delete"])
        async def delete_all_documents(namespace: str = "default"):
            try:
                ids = get_documents_chroma(namespace=namespace)[0]
                if ids:
                    # Chroma doesn't support async
                    get_vector_store(namespace).delete(ids)
                    return JSONResponse({})

            except Exception as ex:
                logger.error(f"Error at delete_all_documents: {ex}\n{traceback.format_exc()}")
                return JSONResponse({"error": "Internal server error"}, 500)

        @app.delete("/document/{namespace}/{id}", tags=["Delete"])
        async def delete_document(id: str, namespace: str = "default"):
            try:
                # Chroma doesn't support async
                get_vector_store(namespace).delete([id])
                return JSONResponse({})

            except Exception as ex:
                logger.error(f"Error at delete_document: {ex}\n{traceback.format_exc()}")
                return JSONResponse({"error": "Internal server error"}, 500)
