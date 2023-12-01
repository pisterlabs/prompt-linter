import os
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.document_loaders import PyMuPDFLoader
from common import document_spliter_len
from qdrant_client import qdrant_client
from hashlib import md5


class QdrantIndex(object):
    """Qdrant index"""

    def __init__(self):
        """Initialize the Qdrant index"""
        self.qdrant_url = os.environ.get("QDRANT_URL")
        self.qdrant_grpc = os.environ.get("QDRANT_GRPC") in ["1", "true", "True", "TRUE"]

    async def search(self, collection, text, topk=3):
        """Search the knowledge base content index"""
        client = qdrant_client.QdrantClient(
            url=self.qdrant_url, prefer_grpc=self.qdrant_grpc
        )
        embeddings = OpenAIEmbeddings()
        q = Qdrant(
            client=client, collection_name=collection,
            content_payload_key="text",
            embeddings=embeddings,
        )
        result = await q.asimilarity_search_with_score(text, k=topk)
        data = []
        if result:
            for doc, score in result:
                data.append(dict(content=doc.page_content, metadata=doc.metadata, score=score))
        return data

    async def delete(self, collection):
        """delete the knowledge base content index"""
        client = qdrant_client.QdrantClient(
            url=self.qdrant_url, prefer_grpc=self.qdrant_grpc
        )
        resp = client.delete_collection(collection_name=collection)
        client.close()
        return resp

    async def list_index(self):
        """delete the knowledge base content index"""
        client = qdrant_client.QdrantClient(
            url=self.qdrant_url, prefer_grpc=self.qdrant_grpc
        )
        resp = client.get_collections()
        client.close()
        return resp

    async def index_text_from_url(self, collection, url, chunk_size=100, chunk_overlap=0):
        """Create a knowledge base content index from web url"""
        loader = WebBaseLoader(url)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap,
                                                       length_function=document_spliter_len)
        docs = text_splitter.split_documents(documents)
        ids = [md5(t.page_content.encode()).hexdigest() for t in docs]
        embeddings = OpenAIEmbeddings()
        await Qdrant.afrom_documents(
            docs, embeddings,
            ids=ids,
            url=self.qdrant_url,
            content_payload_key="text",
            prefer_grpc=self.qdrant_grpc,
            collection_name=collection,
        )

    async def index_pdf_from_path(self, collection, pdffile, chunk_size=1000, chunk_overlap=0):
        """Create a knowledge base content index from pdf file"""
        loader = PyMuPDFLoader(pdffile)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap,
                                                       length_function=document_spliter_len)
        docs = text_splitter.split_documents(documents)
        ids = [md5(t.page_content.encode()).hexdigest() for t in docs]
        embeddings = OpenAIEmbeddings()
        await Qdrant.afrom_documents(
            docs, embeddings,
            ids=ids,
            url=self.qdrant_url,
            content_payload_key="text",
            prefer_grpc=self.qdrant_grpc,
            collection_name=collection,
        )

    async def index_texts(self, collection, texts: list, metadatas: list, chunk_size=2000, chunk_overlap=0):
        """Create a knowledge base content index from text"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=document_spliter_len
        )
        docs = text_splitter.create_documents(texts, metadatas)
        ids = [md5(t.page_content.encode()).hexdigest() for t in docs]
        embeddings = OpenAIEmbeddings()
        await Qdrant.afrom_documents(
            docs, embeddings,
            ids=ids,
            url=self.qdrant_url,
            prefer_grpc=self.qdrant_grpc,
            content_payload_key="text",
            collection_name=collection,
        )


qdrant = QdrantIndex()
