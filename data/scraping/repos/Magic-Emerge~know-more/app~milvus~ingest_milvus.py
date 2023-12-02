"""Load html from files, clean up, split, ingest into Weaviate."""
from typing import List

from langchain.document_loaders import ReadTheDocsLoader, UnstructuredFileLoader, TextLoader, \
    UnstructuredPDFLoader, SeleniumURLLoader, UnstructuredHTMLLoader, \
    UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter

from app.config.conf import MILVUS_CONNECTION_ARGS, DEFAULT_COLLECTION_NAME, DEFAULT_TEXT_FIELD
from app.milvus.milvus import Milvus


def ingest_md_milvus(
        file_path: str,
        collection_name: str,
        text_field: str,
        chunk_size=2000,
        chunk_overlap=200
):
    loader = UnstructuredMarkdownLoader(file_path)
    raw_documents = loader.load()
    text_splitter = MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings(openai_api_version="2020-11-07")
    if Milvus.exist_collection(connection_args=MILVUS_CONNECTION_ARGS, collection_name=collection_name):
        milvus = Milvus.from_existing(embedding=embeddings, connection_args=MILVUS_CONNECTION_ARGS,
                                      collection_name=collection_name, text_field=text_field)
        milvus.add_documents(documents)
    else:
        Milvus.from_documents(documents, embeddings,
                              connection_args=MILVUS_CONNECTION_ARGS,
                              collection_name=collection_name,
                              text_field=text_field)


def ingest_docx_2_milvus(
        file_path: str,
        collection_name: str,
        text_field: str,
        chunk_size=2000,
        chunk_overlap=200,
):
    loader = UnstructuredWordDocumentLoader(file_path=file_path, mode="elements")
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings(openai_api_version="2020-11-07")
    if Milvus.exist_collection(connection_args=MILVUS_CONNECTION_ARGS, collection_name=collection_name):
        milvus = Milvus.from_texts(embedding=embeddings, connection_args=MILVUS_CONNECTION_ARGS,
                                   collection_name=collection_name, text_field=text_field)
        milvus.add_documents(documents)
    else:
        Milvus.from_documents(documents, embeddings,
                              connection_args=MILVUS_CONNECTION_ARGS,
                              collection_name=collection_name,
                              text_field=text_field)


def ingest_html_2_milvus(
        file_path: str,
        collection_name: str,
        text_field: str,
        chunk_size=2000,
        chunk_overlap=200,
):
    loader = UnstructuredHTMLLoader(file_path)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings(openai_api_version="2020-11-07")
    if Milvus.exist_collection(connection_args=MILVUS_CONNECTION_ARGS, collection_name=collection_name):
        milvus = Milvus.from_existing(embedding=embeddings, connection_args=MILVUS_CONNECTION_ARGS,
                                      collection_name=collection_name)
        milvus.add_documents(documents)
    else:
        Milvus.from_documents(documents, embeddings,
                              connection_args=MILVUS_CONNECTION_ARGS,
                              collection_name=collection_name,
                              text_field=text_field)


def ingest_url_2_milvus(
        urls: List[str],
        collection_name: str,
        text_field: str,
        chunk_size=2000,
        chunk_overlap=200,
):
    loader = SeleniumURLLoader(urls=urls)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings(openai_api_version="2020-11-07")
    if Milvus.exist_collection(connection_args=MILVUS_CONNECTION_ARGS, collection_name=collection_name):
        milvus = Milvus.from_existing(embedding=embeddings, connection_args=MILVUS_CONNECTION_ARGS,
                                      collection_name=collection_name, text_field=text_field)
        milvus.add_documents(documents)
    else:
        Milvus.from_documents(documents, embeddings,
                              connection_args=MILVUS_CONNECTION_ARGS,
                              collection_name=collection_name,
                              text_field=text_field)


def ingest_pdf_2_milvus(
        file_path: str,
        collection_name: str,
        text_field: str,
        chunk_size=2000,
        chunk_overlap=200,
):
    """
    Get documents from pdf files.
    :param file_path 文件路径
    :param collection_name milvus collection
    :param text_field milvus文本字段名称
    :param chunk_size 文本切分大小，默认2000
    :param chunk_overlap 切分重叠大小，默认200
    """
    loader = UnstructuredFileLoader(file_path=file_path, mode="elements")
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings(openai_api_version="2020-11-07")
    if Milvus.exist_collection(connection_args=MILVUS_CONNECTION_ARGS, collection_name=collection_name):
        milvus = Milvus.from_existing(embedding=embeddings, connection_args=MILVUS_CONNECTION_ARGS,
                                      collection_name=collection_name, text_field=text_field)
        milvus.add_documents(documents)
    else:
        Milvus.from_documents(documents, embeddings,
                              connection_args=MILVUS_CONNECTION_ARGS,
                              collection_name=collection_name,
                              text_field=text_field)


def ingest_txt_2_milvus(
        file_path: str,
        collection_name: str,
        text_field: str,
        chunk_size=2000,
        chunk_overlap=200,
):
    """Get documents from web pages."""
    loader = UnstructuredHTMLLoader(file_path, encoding='UTF-8')
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings(openai_api_version="2020-11-07")

    if Milvus.exist_collection(connection_args=MILVUS_CONNECTION_ARGS, collection_name=collection_name):
        milvus = Milvus.from_existing(embedding=embeddings, connection_args=MILVUS_CONNECTION_ARGS,
                                      collection_name=collection_name)
        milvus.add_documents(documents)
    else:
        Milvus.from_documents(documents, embeddings,
                              connection_args=MILVUS_CONNECTION_ARGS,
                              collection_name=collection_name)

    # if milvus:
    #     milvus.add_documents(documents)


def ingest_docs_2_milvus(
        file_path: str,
        collection_name: str,
        text_field: str,
        chunk_size=2000,
        chunk_overlap=200,
):
    """Get documents from web pages."""
    loader = UnstructuredHTMLLoader(file_path, encoding='UTF-8')
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings(openai_api_version="2020-11-07")
    vectorstore = Milvus.from_documents(documents, embeddings,
                                        connection_args=MILVUS_CONNECTION_ARGS,
                                        collection_name=collection_name,
                                        text_field=text_field)


if __name__ == '__main__':
    ingest_txt_2_milvus(file_path="../../assets/templates/badcase.txt", collection_name=DEFAULT_COLLECTION_NAME,
                        text_field=DEFAULT_TEXT_FIELD)
