from langchain.vectorstores import qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os

# Create a Qdrant client

os.environ[
    "QDRANT_HOST"
] = "https://97ee601e-c15f-470f-8ebf-0c3a27e1602f.us-east-1-0.aws.cloud.qdrant.io:6333"
os.environ["QDRANT_API_KEY"] = "AxULpTPqfPPXw2tTMg0NROq4wn97y59u7Muwb8jBcp7fHrQ0ONz7Lw"

client = qdrant_client.QdrantClient(
    os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Create a Qdrant collection
os.environ["QDRANT_COLLECTION_NAME"] = "project-wiki-1"

vectors_config = drant_client.http.models.VectorParams(
    size=100,
    distance=models.Distance.COSINE
)

client.recreate_collection(
    collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
    vectors_config=,
)
# restart at 15:25 https://www.youtube.com/watch?v=VL6MAAgwSDM
