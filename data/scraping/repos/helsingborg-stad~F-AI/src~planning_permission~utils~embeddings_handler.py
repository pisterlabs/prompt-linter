from abc import ABC, abstractmethod
import openai


class AbstractEmbeddingsGenerator(ABC):
    """
    Abstract base class for generating embeddings. This class requires
    subclasses to implement the `create_embeddings` method.

    Methods
    -------
    create_embeddings(query: Any) -> Any
        This method must be implemented by subclasses to generate embeddings
        from the given query.
    """

    @abstractmethod
    def create_embeddings(self, query) -> list[float]:
        """
        Abstract method that must be implemented by subclasses to generate
        embeddings from the given query.

        Parameters
        ----------
        query : Any
            The input data for generating the embeddings.

        Returns
        -------
        Any
            The generated embeddings.
        """
        pass


class OpenAIGenerator(AbstractEmbeddingsGenerator):
    """
    Class for generating embeddings using OpenAI's API.

    Methods
    -------
    create_embeddings(query: str) -> list[float]
        Generate embeddings for the given query using OpenAI's API.

    Inherits
    --------
    AbstractEmbeddingsGenerator
    """

    def create_embeddings(self, query) -> list[float]:
        query_as_embedding = (
            openai.Embedding.create(model="text-embedding-ada-002", input=query)
            .data[0]  # type: ignore
            .embedding
        )

        return query_as_embedding


class Embeddings:
    """
    Class for generating embeddings from chunks of text using an instance of
    `AbstractEmbeddingsGenerator`.

    Methods
    -------
    from_chunks(generator: AbstractEmbeddingsGenerator, chunks: list) -> list
        Generate embeddings for each chunk in the list of chunks using the
        provided generator.
    """

    def from_chunks(self, generator: AbstractEmbeddingsGenerator, chunks: list) -> list:
        embeddings = []
        for index, chunk in enumerate(chunks):
            print(f"Generating embeddings {index + 1}/{len(chunks)}", end="\r")  # TODO: Remove side effect print
            result = generator.create_embeddings(query=chunk)
            embeddings.append(result)

        return embeddings
