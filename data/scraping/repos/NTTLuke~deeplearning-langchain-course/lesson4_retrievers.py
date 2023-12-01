# Retrieval

# MMR (Maximal Marginal Relevance)
# you may not want to choose the most similar responses

# LLM Aided Retrieval
# SelfQuery : where we use an LLM to convert the user question into a query

# Compression
# Comporession LLM and then pass the result to LLM


# The LangChain retriever abstraction includes other ways to retrieve documents, such as TF-IDF or SVM using NLP techniques.

import os, sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()


from langchain.vectorstores import Chroma
from llms.llm import azure_openai_embeddings, azure_chat_openai_llm

persist_directory = "data/chroma/"


# Similarity Search Technique
def metadata_search_technique():
    embedding = azure_openai_embeddings()

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    print(vectordb._collection.count())

    question = "what did they say about regression in the first lecture?"

    # filter by metadata
    docs = vectordb.similarity_search(
        question, k=3, filter={"source": "./documents/MachineLearning-Lecture01.pdf"}
    )
    print(docs)


# MMR technique , get better results and avoid redundancy and duplication
def different_search_techniques():
    embedding = azure_openai_embeddings()

    # Load texts into the database
    texts = [
        """The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).""",
        """A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.""",
        """A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.""",
    ]
    smalldb = Chroma.from_texts(texts, embedding=embedding)

    question = "Tell me about all-white mushrooms with large fruiting bodies"
    # Similarity search technique
    # result_similarity = smalldb.similarity_search(question, k=2)
    # print(result_similarity)

    result_relevance = smalldb.max_marginal_relevance_search(question, k=2, fetch_k=3)
    print(result_relevance)


# LLM Aided Retrieval
# But we have an interesting challenge: we often want to infer the metadata from the query itself.
# To address this, we can use SelfQueryRetriever, which uses an LLM to extract:
# - The query string to use for vector search
# - A metadata filter to pass in as well
def llm_search_technique():
    from langchain.retrievers.self_query.base import SelfQueryRetriever
    from langchain.chains.query_constructor.base import AttributeInfo

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The lecture the chunk is from, should be one of `./documents/MachineLearning-Lecture01.pdf`, `./documents/MachineLearning-Lecture02.pdf`, or `./documents/MachineLearning-Lecture03.pdf`",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="The page from the lecture",
            type="integer",
        ),
    ]

    llm = azure_chat_openai_llm()
    embedding = azure_openai_embeddings()

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    document_content_description = "Lecture notes"
    retriever = SelfQueryRetriever.from_llm(
        llm, vectordb, document_content_description, metadata_field_info, verbose=True
    )

    question = "what did they say about regression in the first lecture?"

    docs = retriever.get_relevant_documents(question)
    for d in docs:
        print(d)


def compression_search_technique():
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor

    # Wrap our vectorstore
    llm = azure_chat_openai_llm()
    embedding = azure_openai_embeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    compressor = LLMChainExtractor.from_llm(llm)

    # similarity search and the result is redundant and duplicated
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, base_retriever=vectordb.as_retriever()
    # )

    # MMR search and the result is more relevant
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectordb.as_retriever(search_type="mmr"),
    )

    question = "what did they say about matlab?"
    compressed_docs = compression_retriever.get_relevant_documents(question)
    pretty_print_docs(compressed_docs)


def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


compression_search_technique()
