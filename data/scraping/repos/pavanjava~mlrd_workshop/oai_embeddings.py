from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())


class OAIEmbeddingExtractor:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def embeddings(self, text):
        response = self.client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
            encoding_format="float"
        )
        if response.data is not None:
            return response.data[0].embedding


if __name__ == "__main__":
    obj = OAIEmbeddingExtractor()
    print(obj.embeddings("I love Data Science, AI and Generative AI"))
