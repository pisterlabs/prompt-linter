import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

from notion_utils.MyNotionDBLoader import MyNotionDBLoader

from notion_load.load_util import split_documents


def get_qdrant_client(url: str,
                      api_key: str
                      ) -> QdrantClient:
    qclient = QdrantClient(
        url=url,
        prefer_grpc=True,
        api_key=api_key,
    )
    return qclient


def get_vector_db(q_client: QdrantClient,
                  collection_name: str,
                  embeddings
                  ) -> Qdrant:
    vectors = Qdrant(
        client=q_client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    return vectors


def recreate_collection(q_client: QdrantClient) -> None:
    q_client.recreate_collection(
        # https://qdrant.tech/documentation/how_to/#prefer-high-precision-with-high-speed-search
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
        optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8,
                always_ram=True,
            ),
        ),
    )


def load_qdrant(args):
    """Fetch documents from Notion"""
    notion_loader = MyNotionDBLoader(
        os.getenv("NOTION_TOKEN"),
        os.getenv("NOTION_DATABASE_ID"),
        args.verbose,
        metadata_filter_list=['id', 'title', 'tags', 'version', 'source id', 'published', 'source', 'myid'],
    )
    docs = notion_loader.load()
    print(f"\nFetched {len(docs)} documents from Notion")

    """Split documents into chunks"""
    doc_chunks = split_documents(docs)

    """Get Qdrant client"""
    q_client = get_qdrant_client(os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY"))

    if args.reset:
        print("\nStart recreating Qdrant collection...")
        recreate_collection(q_client)
        print("Finished recreating Qdrant collection")

    if args.verbose:
        collection_info = q_client.get_collection(collection_name=os.getenv("QDRANT_COLLECTION_NAME"))
        print(f"\nCollection info: {collection_info.json()}")

    """Qdrant Vector DB"""
    embeddings = OpenAIEmbeddings()
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    vectors = get_vector_db(q_client, collection_name, embeddings)

    print("\nStart loading documents to Qdrant...")

    batch_chunk_size = 50
    print(f"Number of documents: {len(doc_chunks)}")
    doc_chunks_list = [doc_chunks[i:i + batch_chunk_size] for i in range(0, len(doc_chunks), batch_chunk_size)]
    number_of_batches = len(doc_chunks_list)
    print(f"Number of batches: {number_of_batches}")

    for j in range(0, len(doc_chunks_list)):
        print(f"Loading batch number {j + 1} of {number_of_batches}...")

        Qdrant.add_documents(
            self=vectors,
            documents=doc_chunks_list[j],
        )

    print("Finished loading documents to Qdrant")

    if args.verbose:
        collection_info = q_client.get_collection(collection_name=os.getenv("QDRANT_COLLECTION_NAME"))
        print(f"\nCollection info: {collection_info.json()}")
