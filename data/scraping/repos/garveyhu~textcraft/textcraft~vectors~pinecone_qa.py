import pinecone
from langchain.chains import RetrievalQA
from langchain.vectorstores.pinecone import Pinecone
from textcraft.core.config import keys_pinecone

from textcraft.models.embeddings.embedding_creator import EmbeddingCreator
from textcraft.models.llms.llm_creator import LLMCreator


def vector_qa(question: str) -> str:
    PINECONE_ENV, PINECONE_API_KEY = keys_pinecone()

    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    embeddings = EmbeddingCreator.create_embedding()
    llm = LLMCreator.create_llm()
    docsearch = Pinecone.from_existing_index("langchain", embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    result = qa({"query": question})

    return result.get("result", "")


if __name__ == "__main__":
    print(vector_qa("..."))
