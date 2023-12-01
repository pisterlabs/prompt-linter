import os
import openai


class BaseEmbedding:
    def __init__(self, model: str = "text-embedding-ada-002"):
        self.model = model
        openai.api_key = os.environ["OPENAI_API_KEY"]

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=self.model)["data"][0][
            "embedding"
        ]
