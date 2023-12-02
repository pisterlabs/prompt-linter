import requests, json
from typing import List, Dict, Any, Optional
from langchain.embeddings.base import Embeddings

from settings import logging_config
import logging.config

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)


class InferenceEndpointHuggingFaceEmbeddings(Embeddings):
    def __init__(self, endpoint_name=None, api_token=None) -> None:
        self.endpoint_name = endpoint_name
        self.api_token = api_token

    def embed_query(self, text: str) -> List[float]:
        """Query Hugging Face Inference Endpoint for embeddings

        Args:
            inputs (List[str]): List of texts to embed
        Raises:
            e: Error connecting to Hugging Face API
            Exception: error querying Hugging Face API

        Returns:
            List[List[float]]: List of embeddings
        """
        headers: Dict[str, str] = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
        logger.info(f"Querying Hugging Face API for {text} inputs:")
        query = json.dumps({"inputs": text})
        try:
            response = requests.request("POST", self.endpoint_name, headers=headers, data=query)
        except Exception as e:
            logger.error(f"Error querying Hugging Face API: {e}")
            raise e

        if response.status_code != 200:
            logger.error(f"Error querying Hugging Face API: {response.text}")
            raise Exception(response.text)
        return json.loads(response.content.decode("utf-8"))["embeddings"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Query Hugging Face Inference Endpoint for multiple embeddings

        Args:
            inputs (List[str]): List of texts to embed
        Raises:
            e: Error connecting to Hugging Face API
            Exception: error querying Hugging Face API

        Returns:
            List[List[float]]: List of embeddings
        """
        headers: Dict[str, str] = {"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"}
        logger.info(f"Querying Hugging Face API for {len(texts)} inputs:")
        query = json.dumps({"inputs": texts})
        try:
            response = requests.request("POST", self.endpoint_name, headers=headers, data=query)
        except Exception as e:
            logger.error(f"Error querying Hugging Face API: {e}")
            raise e

        if response.status_code != 200:
            logger.error(f"Error querying Hugging Face API: {response.text}")
            raise Exception(response.text)
        return json.loads(response.content.decode("utf-8"))["embeddings"]


if __name__ == "__main__":
    from config import HUGGING_FACE_EMBEDDINGS_ENDPOINT, HUGGING_FACE_API_TOKEN

    embedding_model = InferenceEndpointHuggingFaceEmbeddings(HUGGING_FACE_EMBEDDINGS_ENDPOINT, HUGGING_FACE_API_TOKEN)
    output = embedding_model.embed_query("What is rainfall data?")
    assert len(output) == 768
    # print(output)
    multiple = embedding_model.embed_documents(["What is rainfall data?", "What is temperature data?"])
    # print(multiple)
    assert len(multiple) == 2
