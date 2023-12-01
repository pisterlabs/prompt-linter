import faiss
from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import FAISS


class VectorStoreMemoryWrapper:
    """ Helper class that wraps around the vector store memory code so we can abstract away the complexity
    of building it while still hjaving access to visualize the memory later.
    """

    def __init__(self) -> None:
        self.embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
        self.index = faiss.IndexFlatL2(self.embedding_size)
        self.embedding_fn = OpenAIEmbeddings().embed_query
        self.vectorstore = FAISS(self.embedding_fn, self.index, InMemoryDocstore({}), {})

    def build_vector_store_retrieval_memory(self, num_vectors_to_load_into_context=10):
        """Builds and returns the memory variable to be passed to an LLM

        Returns:
            memory (VectorStoreRetrieverMemory): the memory class 
        """
        self.retriever = self.vectorstore.as_retriever(search_kwargs=dict(k=num_vectors_to_load_into_context))
        self.memory = VectorStoreRetrieverMemory(retriever=self.retriever)

        return self.memory


def main():
    VSMemory = VectorStoreMemoryWrapper()
    memory = VSMemory.build_vector_store_retrieval_memory()


if __name__ == 'main':
    main()
