import os
import chromadb
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever

# Get 3 diff embeddings.
all_mini = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
multi_qa_mini = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-dot-v1")
filter_embeddings = OpenAIEmbeddings()

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "db")

# Instantiate 2 diff cromadb indexs, each one with a diff embedding.
client_settings = chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=DB_DIR,
    anonymized_telemetry=False,
)
db_all = Chroma(
    collection_name="project_store_all",
    persist_directory=DB_DIR,
    client_settings=client_settings,
    embedding_function=all_mini,
)
db_multi_qa = Chroma(
    collection_name="project_store_multi",
    persist_directory=DB_DIR,
    client_settings=client_settings,
    embedding_function=multi_qa_mini,
)

# Define 2 diff retrievers with 2 diff embeddings and diff search type.
retriever_all = db_all.as_retriever(
    search_type="similarity", search_kwargs={"k": 5, "include_metadata": True}
)
retriever_multi_qa = db_multi_qa.as_retriever(
    search_type="mmr", search_kwargs={"k": 5, "include_metadata": True}
)

# The Lord of the Retrievers will hold the ouput of boths retrievers and can be used as any other
# retriever on different types of chains.
lotr = MergerRetriever(retrievers=[retriever_all, retriever_multi_qa])

