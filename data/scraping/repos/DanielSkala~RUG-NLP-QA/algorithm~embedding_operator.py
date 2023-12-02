from abc import ABC, abstractmethod
from algorithm.models import TextEntry, EmbeddingEntry
from sentence_transformers import SentenceTransformer
from openai import Embedding as OpenAIEmbedding
import openai
import os
from tqdm import tqdm


class EmbeddingOperator(ABC):
    @abstractmethod
    def embed(self, entries: [TextEntry], *args, **kwargs) -> [EmbeddingEntry]:
        pass


class ModelEmbeddingOperator(EmbeddingOperator):

    def __init__(self, model_name: str):
        self.model_name = model_name

    def embed(self, entries: [TextEntry], *args, **kwargs) -> [EmbeddingEntry]:
        model = SentenceTransformer(self.model_name)

        texts = [entry.text for entry in entries]
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

        return [
            EmbeddingEntry(
                id=entry.id,
                embedding=list(embedding),
                metadata=entry.metadata
            ) for entry, embedding in
            zip(entries, embeddings)]


class OpenAIEmbeddingOperator(EmbeddingOperator):

    def __init__(self, model_name: str):
        self.model_name = model_name
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def embed(self, entries: [TextEntry], *args, **kwargs) -> [EmbeddingEntry]:
        embeddings = OpenAIEmbedding.create(
            model=self.model_name,
            input=[entry.text for entry in entries]
        )["data"]
        embeddings = [embedding["embedding"] for embedding in embeddings]
        return [
            EmbeddingEntry(
                id=entry.id,
                embedding=embedding,
                metadata=entry.metadata
            ) for entry, embedding in
            zip(entries, embeddings)]


if __name__ == '__main__':
    # operator = ModelEmbeddingOperator('../artifacts/distiluse-base-multilingual-cased-v1')
    # entries = [
    #     TextEntry(
    #         id="1",
    #         text='hello world',
    #         metadata={}
    #     ),
    #     TextEntry(
    #         id="2",
    #         text='gay',
    #         metadata={}
    #     ),
    # ]
    #
    # embeddings = operator.embed(entries)
    # print(embeddings)

    operator = OpenAIEmbeddingOperator('text-embedding-ada-002')
    entries = [
        TextEntry(
            id="1",
            text='hello world',
            metadata={}
        ),
        TextEntry(
            id="2",
            text='gay',
            metadata={}
        ),
    ]

    embeddings = operator.embed(entries)
    print(embeddings)

