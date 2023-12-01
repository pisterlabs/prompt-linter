import os
from pymilvus import Collection, connections
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from utils import (
    get_abs_path,
    convert_to_2d
)
from config.global_config import (
    OPENAI_API_KEY,
    OPENAI_API_BASE,
    ZILLIZ_CLOUD_URI,
    ZILLIZ_CLOUD_USERNAME,
    ZILLIZ_CLOUD_PASSWORD,
    ZILLIZ_CLOUD_COLLECTION_NAME
)

source_folder = get_abs_path('laws')


def add_embeddings(fold_path):
    loader = DirectoryLoader(fold_path, glob='**/*.md')
    docs = loader.load()
    print(f"{fold_path} 已成功加载")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    # 切割加载的 document
    print("start split docs...")
    split_docs = text_splitter.split_documents(docs)
    print("split docs finished")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,
                                  openai_api_base=OPENAI_API_BASE,
                                  show_progress_bar=True)
    vector_db = Milvus(
        embeddings,
        collection_name=ZILLIZ_CLOUD_COLLECTION_NAME,
        connection_args={
            "uri": ZILLIZ_CLOUD_URI,
            "user": ZILLIZ_CLOUD_USERNAME,
            "password": ZILLIZ_CLOUD_PASSWORD,
            "secure": True,
        },
    )

    convert_docs = convert_to_2d(split_docs, 8)
    convert_docs_len = len(convert_docs)
    for index, docs in enumerate(convert_docs):
        try:
            vector_db.add_documents(docs)
            print(f"init vector index group {index + 1} finished, remain {convert_docs_len - index - 1}")
        except Exception as e:
            print(f"init vector index error: {e}")
            source_name = [os.path.split(doc.metadata['source'])[1] for doc in docs]
            print(','.join(set(source_name)))


def get_collection_detail():
    connections.connect(
        uri=ZILLIZ_CLOUD_URI,
        user=ZILLIZ_CLOUD_USERNAME,
        password=ZILLIZ_CLOUD_PASSWORD,
        port='19530'
    )
    collection = Collection(ZILLIZ_CLOUD_COLLECTION_NAME)
    print(collection)
    print(collection.num_entities)


get_collection_detail()
# add_embeddings(source_folder)
