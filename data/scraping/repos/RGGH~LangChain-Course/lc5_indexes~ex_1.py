# LangChai Imports
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Qdrant Imports
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import CollectionStatus
from qdrant_client.models import PointStruct
from qdrant_client.models import Distance, VectorParams

#
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm

# Create a constant for our collection
COLLECTION_NAME = "aiw"

# Eventual list of books
TEXTS = ["/home/rag/Documents/python/course/lc5_indexes/text/aiw.txt"]

vectors = []
batch_size = 512
batch = []

model = SentenceTransformer(
    "msmarco-MiniLM-L-6-v3",
)

# Client
client = QdrantClient(host="localhost", port=6333, prefer_grpc=False)


def make_collection(client, collection_name: str):
    """
    Use this the 1st time you run the code
    """

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )


def make_chunks(inptext: str):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n", chunk_size=1000, chunk_overlap=20, length_function=len
    )

    with open(inptext) as f:
        alice = f.read()

    chunks = text_splitter.create_documents([alice])

    return chunks


texts = make_chunks(TEXTS[0])


# Create the VECTORS
def gen_vectors(texts, model, batch, batch_size, vectors):
    for part in tqdm(texts):
        batch.append(part.page_content)

        if len(batch) >= batch_size:
            vectors.append(model.encode(batch))
            batch = []

    if len(batch) > 0:
        vectors.append(model.encode(batch))
        batch = []

    vectors = np.concatenate(vectors)

    payload = [item for item in texts]
    payload = list(payload)
    vectors = [v.tolist() for v in vectors]
    return vectors, payload


fin_vectors, fin_payload = gen_vectors(
    texts=texts, model=model, batch=batch, batch_size=batch_size, vectors=vectors
)


# Upsert
def upsert_to_qdrant(fin_vectors, fin_payload):
    collection_info = client.get_collection(collection_name=COLLECTION_NAME)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=collection_info.vectors_count + idx, vector=vector, payload=fin_payload[idx]
            )
            for idx, vector in enumerate(fin_vectors)
        ],
    )


# run once
make_collection(client, "aiw")

# perform the upsert!
upsert_to_qdrant(fin_vectors, fin_payload)
