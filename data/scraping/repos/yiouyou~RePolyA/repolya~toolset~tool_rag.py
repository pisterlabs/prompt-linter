from langchain.tools import tool
from langchain.tools import StructuredTool

from repolya.rag.doc_loader import get_docs_from_pdf
from repolya.rag.doc_splitter import split_pdf_docs_recursive
from repolya.rag.vdb_faiss import (
    get_faiss_OpenAI,
    get_faiss_HuggingFace,
)
from repolya.rag.retriever import (
    get_vdb_multi_query_retriever,
    get_docs_ensemble_retriever,
    get_docs_parent_retriever,
)
from repolya.rag.qa_chain import (
    qa_vdb_multi_query,
    qa_docs_ensemble_query,
    qa_docs_parent_query,
)

import os


def rag_vdb_multi_query(_query: str, _db_name: str, _chain_type: str) -> str:
    if 'openai' in _db_name:
        _vdb = get_faiss_OpenAI(_db_name)
    else:
        _vdb = get_faiss_HuggingFace(_db_name)
    _ans, _steps, _token_cost = qa_vdb_multi_query(_query, _vdb, _chain_type)
    return _ans

def tool_rag_vdb_multi_query(_db_name: str):
    _info_type = '_'.join(_db_name.split('_')[:-1])
    tool = StructuredTool.from_function(
        rag_vdb_multi_query,
        name=f"QA {_info_type} with 'MultiQuery' method",
        description=f"For comprehensive querying of {_info_type}, use MultiQueryRetriever; it automates prompt tuning with LLMs, overcoming traditional distance-based method limitations.",
        verbose=True,
    )
    return tool


def rag_pdf_ensemble_query(_query: str, _pdf: str, _chain_type: str) -> str:
    _docs = get_docs_from_pdf(_pdf)
    text_chunk_size = 1000
    text_chunk_overlap = 50
    _splited_docs = split_pdf_docs_recursive(_docs, text_chunk_size, text_chunk_overlap)
    _ans, _steps, _token_cost = qa_docs_ensemble_query(_query, _splited_docs, _chain_type)
    return _ans

def tool_rag_pdf_ensemble_query(_pdf: str):
    _info_type = '.'.join(os.path.basename(_pdf).split('.')[:-1])
    tool = StructuredTool.from_function(
        rag_pdf_ensemble_query,
        name=f"QA {_info_type} with 'Ensemble' method",
        description=f"For comprehensive querying of {_info_type}, use EnsembleRetriever; it blends results from multiple retrievers, merging keyword-centric and semantic search methods for a hybrid approach.",
        verbose=True,
    )
    return tool


def rag_pdf_parent_query(_query: str, _pdf: str, _chain_type: str) -> str:
    _docs = get_docs_from_pdf(_pdf)
    text_chunk_size = 1000
    text_chunk_overlap = 50
    _splited_docs = split_pdf_docs_recursive(_docs, text_chunk_size, text_chunk_overlap)
    _ans, _steps, _token_cost = qa_docs_parent_query(_query, _splited_docs, _chain_type)
    return _ans

def tool_rag_pdf_parent_query(_pdf: str):
    _info_type = '.'.join(os.path.basename(_pdf).split('.')[:-1])
    tool = StructuredTool.from_function(
        rag_pdf_parent_query,
        name=f"QA {_info_type} with 'ParentDocument' method",
        description=f"For comprehensive querying of {_info_type}, use ParentDocumentRetriever; it retrieves small chunks and their originating parent documents, maintaining both accuracy and context.",
        verbose=True,
    )
    return tool

