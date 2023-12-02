from repolya._log import logger_rag

from repolya.rag.doc_loader import get_docs_from_pdf
from repolya.rag.doc_splitter import split_pdf_docs_recursive
from repolya.rag.embedding import get_embedding_OpenAI, get_embedding_HuggingFace

from langchain.vectorstores import FAISS

import pandas as pd
pd.set_option('display.max_rows', 50)

import os


def show_faiss(_vdb):
    _vdb_df = faiss_to_df(_vdb)
    print(_vdb_df)


def faiss_to_df(_vdb):
    v_dict = _vdb.docstore._dict
    _rows = []
    for k in v_dict.keys():
        content = v_dict[k].page_content
        doc_name = v_dict[k].metadata['source'].split('/')[-1] if 'source' in v_dict[k].metadata.keys() else ''
        _rows.append({"chunk_id": k, "doc_name": doc_name, "content": content})
    _vdb_df = pd.DataFrame(_rows)
    return _vdb_df


##### OpenAI
def get_faiss_OpenAI(_db_name):
    _model_name, _embedding = get_embedding_OpenAI()
    _vdb_name = os.path.join(_db_name, _model_name)
    if os.path.exists(_vdb_name):
        _vdb = FAISS.load_local(_vdb_name, _embedding)
        logger_rag.info(f"load {_model_name} embedding from faiss {_vdb_name}")
        return _vdb
    else:
        logger_rag.info(f"NO {_vdb_name}")
        return None


def embedding_to_faiss_OpenAI(_docs, _db_name):
    _model_name, _embedding = get_embedding_OpenAI()
    _vdb_name = os.path.join(_db_name, _model_name)
    if not os.path.exists(_vdb_name):
        _db = FAISS.from_documents(_docs, _embedding)
        _db.save_local(_vdb_name)
        logger_rag.info("/".join(_vdb_name.split("/")[-2:]))
    ### log
    logger_rag.info(f"save {_model_name} embedding to faiss {_vdb_name}")


def pdf_to_faiss_OpenAI(_fp, _db_name, text_chunk_size=3000, text_chunk_overlap=300):
    docs = get_docs_from_pdf(_fp)
    if len(docs) > 0:
        logger_rag.info(f"docs: {len(docs)}")
        splited_docs = split_pdf_docs_recursive(docs, text_chunk_size, text_chunk_overlap)
        logger_rag.info(f"splited_docs: {len(splited_docs)}")
        embedding_to_faiss_OpenAI(splited_docs, _db_name)
    else:
        logger_rag.info("NO docs")


def merge_faiss_OpenAI(_old_name, _add_name):
    ### merge
    _vdb_old = get_faiss_OpenAI(_old_name)
    _vdb_add = get_faiss_OpenAI(_add_name)
    # print(type(_vdb_old))
    # print(type(_vdb_add))
    if _vdb_old is not None:
        _vdb_old.merge_from(_vdb_add)
        logger_rag.info(f"merge '{_add_name.split('/')[-1]}' to '{_old_name.split('/')[-1]}'")
    else:
        _vdb_old = _vdb_add
        logger_rag.info(f"save '{_add_name.split('/')[-1]}' to '{_old_name.split('/')[-1]}'")
    ### save
    _model_name, _embedding = get_embedding_OpenAI()
    _vdb_name = os.path.join(_old_name, _model_name)
    _vdb_old.save_local(_vdb_name)
    _vdb = get_faiss_OpenAI(_old_name)
    show_faiss(_vdb)


def delete_doc_from_faiss_OpenAI(_db_name, _doc_name):
    _vdb = get_faiss_OpenAI(_db_name)
    _vdb_df = faiss_to_df(_vdb)
    chunks_list = _vdb_df.loc[_vdb_df['doc_name'] == _doc_name]['chunk_id'].tolist()
    print(chunks_list)
    _vdb.delete(chunks_list)
    _model_name, _embedding = get_embedding_OpenAI()
    _vdb_name = os.path.join(_db_name, _model_name)
    _vdb.save_local(_vdb_name)


def add_texts_to_faiss_OpenAI(_texts, _metadatas, _db_name):
    _vdb = get_faiss_OpenAI(_db_name)
    # show_faiss(_vdb)
    _ids = _vdb.add_texts(
        texts=_texts,
        metadatas=_metadatas,
        # ids=_ids,
    )
    # show_faiss(_vdb)
    _model_name, _embedding = get_embedding_OpenAI()
    _vdb_name = os.path.join(_db_name, _model_name)
    _vdb.save_local(_vdb_name)
    return _ids


##### HuggingFace
def get_faiss_HuggingFace(_db_name):
    _model_name, _embedding = get_embedding_HuggingFace()
    _vdb_name = os.path.join(_db_name, _model_name)
    if os.path.exists(_vdb_name):
        _vdb = FAISS.load_local(_vdb_name, _embedding)
        logger_rag.info(f"load {_model_name} embedding from faiss {_vdb_name}")
        return _vdb
    else:
        logger_rag.info(f"NO {_vdb_name}")
        return None


def embedding_to_faiss_HuggingFace(_docs, _db_name):
    _model_name, _embedding = get_embedding_HuggingFace()
    _vdb_name = os.path.join(_db_name, _model_name)
    if not os.path.exists(_vdb_name):
        _vdb = FAISS.from_documents(_docs, _embedding)
        _vdb.save_local(_vdb_name)
        logger_rag.info("/".join(_vdb_name.split("/")[-2:]))
    ### log
    logger_rag.info(f"save {_model_name} embedding to faiss {_vdb_name}")


def pdf_to_faiss_HuggingFace(_fp, _db_name, text_chunk_size=3000, text_chunk_overlap=300):
    docs = get_docs_from_pdf(_fp)
    if len(docs) > 0:
        logger_rag.info(f"docs: {len(docs)}")
        splited_docs = split_pdf_docs_recursive(docs, text_chunk_size, text_chunk_overlap)
        logger_rag.info(f"splited_docs: {len(splited_docs)}")
        embedding_to_faiss_HuggingFace(splited_docs, _db_name)
    else:
        logger_rag.info("NO docs")


def merge_faiss_HuggingFace(_old_name, _add_name):
    ### merge
    _vdb_old = get_faiss_HuggingFace(_old_name)
    _vdb_add = get_faiss_HuggingFace(_add_name)
    # print(type(_vdb_old))
    # print(type(_vdb_add))
    if _vdb_old is not None:
        _vdb_old.merge_from(_vdb_add)
        logger_rag.info(f"merge '{_add_name.split('/')[-1]}' to '{_old_name.split('/')[-1]}'")
    else:
        _vdb_old = _vdb_add
        logger_rag.info(f"save '{_add_name.split('/')[-1]}' to '{_old_name.split('/')[-1]}'")
    ### save
    _model_name, _embedding = get_embedding_HuggingFace()
    _vdb_name = os.path.join(_old_name, _model_name)
    _vdb_old.save_local(_vdb_name)
    _vdb = get_faiss_HuggingFace(_old_name)
    show_faiss(_vdb)


def delete_doc_from_faiss_HuggingFace(_db_name, _doc_name):
    _vdb = get_faiss_HuggingFace(_db_name)
    _vdb_df = faiss_to_df(_vdb)
    chunks_list = _vdb_df.loc[_vdb_df['doc_name'] == _doc_name]['chunk_id'].tolist()
    print(chunks_list)
    _vdb.delete(chunks_list)
    _model_name, _embedding = get_embedding_HuggingFace()
    _vdb_name = os.path.join(_db_name, _model_name)
    _vdb.save_local(_vdb_name)


def add_texts_to_faiss_HuggingFace(_texts, _metadatas, _db_name):
    _vdb = get_faiss_HuggingFace(_db_name)
    # show_faiss(_vdb)
    _ids = _vdb.add_texts(
        texts=_texts,
        metadatas=_metadatas,
        # ids=_ids,
    )
    # show_faiss(_vdb)
    _model_name, _embedding = get_embedding_HuggingFace()
    _vdb_name = os.path.join(_db_name, _model_name)
    _vdb.save_local(_vdb_name)
    return _ids

