from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Milvus, VectorStore

text_field = "document_text"


def create_milvus_collection(collection_name: str = "default", recreate: bool = False):
    from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                          connections)

    connections.connect()

    document_id = FieldSchema(
        name="document_id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )
    document_text = FieldSchema(
        name=text_field,
        dtype=DataType.VARCHAR,
        max_length=40000,
    )
    document_embedding = FieldSchema(
        name="document_embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=1536,
    )

    schema = CollectionSchema(
        fields=[document_id, document_text, document_embedding],
        description="Test book search",
    )

    if recreate:
        collection = Collection(name=collection_name)
        collection.drop()
    collection = Collection(name=collection_name, schema=schema)

    return collection


def create_milvus_index(collection_name: str = "default"):
    from pymilvus import Collection, IndexType

    collection = create_milvus_collection(collection_name=collection_name)
    collection.create_index(
        field_name="document_embedding",
        index_params={
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        },
    )


def get_vector_db(collection_name: str = "default") -> VectorStore:
    return Milvus(
        collection_name=collection_name,
        text_field=text_field,
        embedding_function=OpenAIEmbeddings(),
        connection_args={},
    )


def vector_db_wrapper(
    vectorstore: VectorStore | None = None,
) -> VectorStoreIndexWrapper:
    """
    # Example

    >>> from milvus_db import vector_db_wrapper
    >>> vectordb = vector_db_wrapper()
    >>> vectordb.query("What is the airspeed of a fully laden swallow?")

    """
    if vectorstore is None:
        vectorstore = get_vector_db()
    return VectorStoreIndexWrapper(vectorstore=vectorstore)


def add_new_document_to_vector_db(
    vector_db: VectorStoreIndexWrapper,
    document: Document,
):
    vector_db.vectorstore.add_documents([document])


if __name__ == "__main__":
    # create_milvus_collection(recreate=True)
    create_milvus_index()
