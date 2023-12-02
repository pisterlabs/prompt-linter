import os
import requests
from tempfile import NamedTemporaryFile
from celery import Celery
from app import app
from models import save_document, save_embedding
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredFileLoader,
)
from langchain.document_loaders.sitemap import SitemapLoader



def create_celery_app(app=None):
    """
    Create a new Celery object and tie together the Celery config to the app's
    config. Wrap all tasks in the context of the application.

    :param app: Flask app
    :return: Celery app
    """

    app.config['CELERY_BROKER_URL'] = 'redis://redis:6379/0'
    app.config['CELERY_RESULT_BACKEND'] = 'redis://redis:6379/0'
    celery = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'], backend=app.config['CELERY_RESULT_BACKEND'])

    celery.conf.update(app.config.get("CELERY_CONFIG", {}))
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery



celery = create_celery_app(app)


LOADER_MAPPING = {
    "pdf": (PyMuPDFLoader, {}),
    "word": (UnstructuredWordDocumentLoader, {}),
    "excel": (UnstructuredExcelLoader, {}),
    "markdown": (UnstructuredMarkdownLoader, {}),
    "ppt": (UnstructuredPowerPointLoader, {}),
    "txt": (TextLoader, {"encoding": "utf8"}),
    "html": (UnstructuredHTMLLoader, {}),
    "sitemap": (SitemapLoader, {}),
    "default": (UnstructuredFileLoader, {}),
}


def embedding_single_document(doc, fileUrl, fileType, fileName, collection_id, openai=False, uniqid=''):
    # 初始化embeddings
    if openai:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(model_name="/m3e-base")
    # 初始化加载器
    # text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    split_docs = text_splitter.split_documents([doc])
    # 保存documents
    document_id = save_document(collection_id, fileName or fileUrl, fileUrl, len(split_docs), fileType, uniqid=uniqid)
    # document_ids.append(document_id)
    doc_result = embeddings.embed_documents([d.page_content for d in split_docs])
    for chunk_index, doc in enumerate(split_docs):
        save_embedding(
            collection_id, document_id,
            chunk_index, len(doc.page_content),
            doc.page_content,
            doc_result[chunk_index],  # embed
        )
    return document_id


def get_status_by_id(task_id):
    return celery.AsyncResult(task_id)


def embed_query(text, openai=False):
    # 初始化embeddings
    if openai:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(model_name="/m3e-base")

    return embeddings.embed_query(text)


