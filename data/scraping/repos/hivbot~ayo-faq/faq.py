import util as util
import pandas as pd
import os
import shutil
from typing import List, Tuple
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

EMBEDDING_MODEL_FOLDER = ".embedding-model"
VECTORDB_FOLDER = ".vectordb"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def create_documents(df: pd.DataFrame, page_content_column: str) -> pd.DataFrame:
    loader = DataFrameLoader(df, page_content_column=page_content_column)
    return loader.load()


def define_embedding_function(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
        cache_folder=EMBEDDING_MODEL_FOLDER,
    )


def get_vectordb(
    collection_id: str,
    embedding_function: Embeddings,
    documents: List[Document] = None,
) -> VectorStore:
    vectordb = None

    if documents is None:
        try:
            vectordb = FAISS.load_local(VECTORDB_FOLDER+"/"+collection_id, embedding_function)
        except RuntimeError:
            vectordb = None
            raise Exception("collection_id may not exists")
    else:
        vectordb = FAISS.from_documents(
            documents=documents,
            embedding=embedding_function,
        )
        vectordb.save_local(VECTORDB_FOLDER+"/"+collection_id)
    return vectordb


def similarity_search(
    vectordb: VectorStore, query: str, k: int = 3
) -> List[Tuple[Document, float]]:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    return vectordb.similarity_search_with_relevance_scores(query=query, k=k)


def load_vectordb_id(
    collection_id: str,
    page_content_column: str,
    embedding_function_name: str = EMBEDDING_MODEL,
) -> VectorStore:
    embedding_function = define_embedding_function(embedding_function_name)
    vectordb = None
    try:
        vectordb = get_vectordb(
            collection_id=collection_id, embedding_function=embedding_function
        )
    except Exception as e:
        print(e)
        vectordb = create_vectordb_id(
            collection_id, page_content_column, embedding_function
        )

    return vectordb


def create_vectordb_id(
    collection_id: str,
    page_content_column: str,
    embedding_function: HuggingFaceEmbeddings = None,
) -> VectorStore:
    if embedding_function is None:
        embedding_function = define_embedding_function(EMBEDDING_MODEL)

    df = util.read_df(util.xlsx_url(collection_id), page_content_column)
    documents = create_documents(df, page_content_column)
    vectordb = get_vectordb(
        collection_id=collection_id,
        embedding_function=embedding_function,
        documents=documents,
    )
    return vectordb


def load_vectordb(sheet_url: str, page_content_column: str) -> VectorStore:
    return load_vectordb_id(util.get_id(sheet_url), page_content_column)


def delete_vectordb() -> None:
    shutil.rmtree(VECTORDB_FOLDER, ignore_errors=True)


def delete_vectordb_sheet_collection(sheet_url: str) -> None:
    shutil.rmtree(VECTORDB_FOLDER+"/"+util.get_id(sheet_url), ignore_errors=True)
