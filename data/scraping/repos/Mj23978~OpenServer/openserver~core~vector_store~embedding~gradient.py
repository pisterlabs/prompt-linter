from typing import List

from langchain.embeddings import GradientEmbeddings

from .base import BaseEmbedding


class GradientEmbedding(BaseEmbedding):
    def __init__(self, gradient_access_token: str, gradient_workspace_id: str, model="bge-large") -> None:
        self.model: str = model
        self.client = GradientEmbeddings(
            model=self.model,
            gradient_access_token=gradient_access_token,
            gradient_workspace_id=gradient_workspace_id,
        )  # type: ignore

    def get_embeddings(self, texts: List[str]) -> List[List[float]] | dict[str, Exception]:
        try:
            embeds: List[List[float]] = self.client.embed_documents(
                texts=texts)
            return embeds
        except Exception as exception:
            return {"error": exception}

    def get_embedding(self, text: str) -> List[float] | dict[str, Exception]:
        try:
            embeds: List[float] = self.client.embed_query(text=text)
            return embeds
        except Exception as exception:
            return {"error": exception}
