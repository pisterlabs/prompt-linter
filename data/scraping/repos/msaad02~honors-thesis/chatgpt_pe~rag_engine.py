"""
Script that makes class for the RAG engine.

It utilizes semantic search with Chroma and leverages GPT-3.5 from 
OpenAI to answer questions.

Ensure you're familiar with the specifics in Version 2 of the README.md 
and refer to https://python.langchain.com/docs/get_started/introduction.html
for comprehensive documentation on the modules used.
"""

from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from torch.cuda import is_available

class TraditionalRAGEngine:
    def __init__(
            self,
            vectordb_persist_dir = "/home/msaad/workspace/honors-thesis/data_collection/data/noncategorized_chroma/filtered_dataset_v1",
        ):
        self.vectordb_persist_dir = vectordb_persist_dir

        self.embedding_function = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={'device': 'cuda' if is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.vector_db = Chroma(
            persist_directory=self.vectordb_persist_dir,
            embedding_function=self.embedding_function
        )

    def __call__(
            self,
            question: str,
            search_args: dict = {"score_threshold": .8, "k": 3}
        ):
        
        model = RetrievalQA.from_chain_type(
            llm = OpenAI(),
            chain_type = "stuff",
            retriever = self.vector_db.as_retriever(search_kwargs=search_args)
        )
        return model.run(question)