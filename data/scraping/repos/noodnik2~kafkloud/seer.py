import logging

import chromadb

from chromadb.config import Settings

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.document_loaders import TextLoader


class Seer:

    def __init__(self, host="localhost", port="8020"):
        chroma_settings = Settings(chroma_server_host=host, chroma_server_http_port=port,
                                   chroma_api_impl="rest", anonymized_telemetry=False,
                                   persist_directory="chroma_persistence", chroma_db_impl="duckdb+parquet")
        self.logger = logging.getLogger(__name__)
        self.chroma_client = chromadb.Client(chroma_settings)
        self.embeddings = OpenAIEmbeddings()

    def ask(self, questions):
        self.logger.debug(f"received questions({questions})")
        vectordb = Chroma(embedding_function=self.embeddings, client=self.chroma_client)
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(), # consider playing with the "temperature" parameter
            chain_type="stuff",
            retriever=vectordb.as_retriever() # consider using 'search_type = e.g., "similarity" or "mar"' - and 'search_kwargs = e.g., {"k":2}' parameters
        )
        answers = []
        for q in questions:
            self.logger.debug(f"running query: {q}")
            answer = qa.run(q)
            self.logger.debug(f"the answer is: {answer}")
            answers.append(answer)
        return answers

    def load(self, fns):
        self.logger.debug(f"received fns({fns})")
        documents_to_split = []
        for fn in fns:
            self.logger.debug(f"reading({fn})")
            loader = TextLoader(fn)
            for doc in loader.load():
                documents_to_split.append(doc)
        self._split_and_load_documents(documents_to_split)

    def accept(self, texts):
        self.logger.debug(f"received texts({texts})")
        documents_to_split = []
        for text in texts:
            documents_to_split.append(Document(page_content=text))
        self._split_and_load_documents(documents_to_split)

    def _split_and_load_documents(self, documents_to_split):
        self.logger.debug("RecursiveCharacterTextSplitter")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.logger.debug("text_splitter.split_documents")
        documents = text_splitter.split_documents(documents_to_split)
        self.logger.debug("Chroma.from_documents")
        Chroma.from_documents(documents, self.embeddings, client=self.chroma_client)

