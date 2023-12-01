import pandas as pd
import spacy
import nltk
from langchain.text_splitter import SpacyTextSplitter, NLTKTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from pyspark.sql import DataFrame
from functools import partial


def init_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download

        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")


def init_nltk():
    nltk.download("punkt")


def split_documents_udf(iterator, col_name: str = "text"):
    # init_spacy_model()
    init_nltk()
    # text_splitter = SpacyTextSplitter(chunk_size=512, chunk_overlap=50)
    text_splitter = NLTKTextSplitter(chunk_size=512, chunk_overlap=50)
    for pdf in iterator:
        res = []
        for rec in pdf.to_dict("records"):
            docs = text_splitter.split_text(rec[col_name])
            for doc in docs:
                new_rec = {}
                new_rec[col_name] = doc
                for k, v in rec.items():
                    if col_name != k:
                        new_rec[k] = v
                res.append(new_rec)
        yield pd.DataFrame.from_records(res)


def split_documents(df: DataFrame, col_name: str = "text") -> DataFrame:
    _split_documents_udf = partial(split_documents_udf, col_name=col_name)
    return df.mapInPandas(_split_documents_udf, df.schema)


def create_vector_db(
    docs_df: DataFrame,
    value_col_name: str,
    metadata_col_name: str,
    collection_name: str = "collection",
    path: str = "/dbfs/tmp/",
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
):
    hf_embed = HuggingFaceEmbeddings(model_name=embedding_model_name)

    documents = [
        Document(
            page_content=rec[value_col_name],
            metadata={"source": rec[metadata_col_name]},
        )
        for rec in docs_df.toPandas().to_dict("records")
    ]

    db = Chroma.from_documents(
        collection_name=collection_name,
        documents=documents,
        embedding=hf_embed,
        persist_directory=path,
    )
    db.similarity_search("dummy")
    db.persist()
    return db
