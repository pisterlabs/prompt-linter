from uuid import uuid4

from langchain.document_loaders.text import TextLoader
from langchain.schema.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models


from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


MEMORY_PATH = "memory.md"
COLLECTION_NAME = "ai_devs"

# Get embedding for query
embeddings = OpenAIEmbeddings()
query = "Do you know the name of Adam's dog?"
query_embedding = embeddings.embed_query(query)


# Initialize Qdrant client
# localhost can be changed to: os.environ['QDRANT_URL']
client = QdrantClient("localhost", port=6333)
result = client.get_collections()

# Check if collection exists
indexed = next(
    (
        collection
        for collection in result.collections
        if collection.name == COLLECTION_NAME
    ),
    None,
)
print(result)

# Create collection if does not exist
if not indexed:
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1536, distance=models.Distance.COSINE, on_disk=True
        ),
    )

# Index documents if not indexed
collection_info = client.get_collection(COLLECTION_NAME)
if not collection_info.points_count:
    # Read File
    loader = TextLoader(MEMORY_PATH)
    memory = loader.load()

    # Create list of Documents with metadata
    # Is it possible to do this without Document class? We should be able to achieve the same result with
    # List of dicts (metadata), as they contain content as well.
    documents = [
        Document(
            page_content=content,
            metadata={
                "content": content,
                "source": COLLECTION_NAME,
                "uuid": str(uuid4()),
            },
        )
        for content in memory[0].page_content.split("\n\n")
    ]

    # Create embeddings
    points = [
        {
            "id": document.metadata["uuid"],
            "payload": document.metadata,
            "vector": embeddings.embed_documents([document.page_content])[0],
        }
        for document in documents
    ]

    # Prepare embeddings for batch upsert
    ids, vectors, payloads = zip(
        *((point["id"], point["vector"], point["payload"]) for point in points)
    )

    # Index
    client.upsert(
        COLLECTION_NAME,
        points=models.Batch(ids=ids, payloads=payloads, vectors=vectors),
    )

# Create filter for search query
query_filter = models.Filter(
    must=[
        models.FieldCondition(
            key="source",
            match=models.MatchValue(value=COLLECTION_NAME),
        )
    ]
)

# Search for embedded query
search = client.search(
    COLLECTION_NAME, query_vector=query_embedding, limit=1, query_filter=query_filter
)
print(search)
