from langchain.chains.base import Chain
from typing import Dict, List
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import Tool
from paperai.config import *
from qdrant_client import QdrantClient, models
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from paperai.vectordb import DatabaseInterface


class UploadInVectorDB(Chain, DatabaseInterface):
    chain: Chain
    output_key: str = "vector_output" 
    vectordb: DatabaseInterface

    @property
    def input_keys(self) -> List[str]:
        return list(self.chain.input_keys)

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def upload(self, query:str) -> bool:
        path_arr = query.split(" ")
        for each in path_arr:
            if '.pdf' in each:
                files = each.split(".pdf")
                doc_path = f"{files[0]}.pdf".strip()
                # Create your PDF loader
                loader = PyPDFLoader(doc_path)
                # Load the PDF document
                documents = loader.load()
                # Why seperators as param https://github.com/hwchase17/langchain/issues/1663
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50,separators=["\n\n", ""])
                docs = text_splitter.split_documents(documents)
                qdrant_upload = Qdrant.from_documents(
                    docs, self.vectordb.embeddings, 
                    path=db_persistent_path,
                    collection_name=collection_name)
                qdrant_upload = None
                return doc_path
        return None

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        # Queries the database to get the relevant documents for a given query
        query = inputs.get("input_documents", "").lower().strip()
        doc_path= self.upload(query)
        if doc_path:
        # output = self.chain.run(input_documents=[f"Document {doc_path} Uploaded in the path {db_persistent_path}"], question=query)
            return { self.output_key : f"Final Answer: Document {doc_path} Uploaded in the path {db_persistent_path}" }
        return  { self.output_key : None }