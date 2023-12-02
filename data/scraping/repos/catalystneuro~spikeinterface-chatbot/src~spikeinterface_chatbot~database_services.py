from pathlib import Path
from typing import Optional, List
import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Qdrant
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def create_prompt(system_template: Optional[str] = None) -> ChatPromptTemplate:
    """
    An auxiliary function to create prompts for the chatbot.
    Making it explicit for prompt experimentation.
    """

    if system_template is None:
        system_template = """Use the following pieces of context to answer the users question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer. Also, return the output
        in markdown format.
        ----------------
        {context}"""

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    chat_prompt = ChatPromptTemplate.from_messages(messages)

    return chat_prompt


def build_question_and_answer_retriever(verbose: bool = False) -> RetrievalQA:
    """
    Builds a question and answer retriever using the GPT-3.5-turbo model and Qdrant database.

    The function retrieves the Qdrant database, initializes a ChatOpenAI instance with the GPT-3.5-turbo model,
    and sets up a retriever with the database. It creates a chat prompt and initializes a RetrievalQA object with
    the necessary configurations.

    Parameters
    ----------
    verbose : bool, optional, default: False
        If True, the function will print the intermediate steps.

    Returns
    -------
    RetrievalQA
        A Retrieval chain with the necessary configurations.
    """

    model_name = "gpt-3.5-turbo"  # Cheaper than default davinci
    max_tokens = None  # Exposed to experiment with different values
    temperature = 0.0
    vectorstore = retrieve_qdrant_database()
    llm = ChatOpenAI(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
    retriever = vectorstore.as_retriever()
    chat_prompt = create_prompt()
    return_source_documents = True  # Returns the source documents from the datbase for the context
    chain_type_kwargs = dict(prompt=chat_prompt)
    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type_kwargs=chain_type_kwargs,
        retriever=retriever,
        return_source_documents=return_source_documents,
        verbose=verbose,
    )

    return retrieval_qa_chain


def retrieve_qdrant_database(
    port: Optional[int] = 6333, collection_name: Optional[str] = "spikeinterface_documentation"
) -> Qdrant:
    """
    Retrieve a Qdrant database by connecting to the Qdrant instance and returning an initialized Qdrant object.

    Parameters
    ----------
    port : Optional[int], default=6333
        The port to use when connecting to the Qdrant instance.
    collection_name : Optional[str], default="spikeinterface_documentation"
        The name of the Qdrant collection to retrieve.

    Returns
    -------
    Qdrant
        An initialized Qdrant object connected to the specified collection.
    """
    import qdrant_client
    from qdrant_client.http.models import CollectionStatus

    url = "https://443e1d88-95ec-48b6-b66d-b5c4713ede35.us-east-1-0.aws.cloud.qdrant.io"
    api_key = os.environ["QDRANT_API_KEY"]
    client = qdrant_client.QdrantClient(url=url, api_key=api_key, port=port)
    collection_info = client.get_collection(collection_name=collection_name)
    assert collection_info.status == CollectionStatus.GREEN

    embedding_function = OpenAIEmbeddings().embed_query
    qdrant = Qdrant(client=client, embedding_function=embedding_function, collection_name=collection_name)

    return qdrant


def generate_qdrant_database_from_docs(
    documents: List[Document],
    port: Optional[int] = 6333,
    collection_name: Optional[str] = "spikeinterface_documentation",
) -> Qdrant:
    """
    Generate a Qdrant database from a list of documents and return an initialized Qdrant object.

    Parameters
    ----------
    documents : List[Document]
        The list of documents to generate the Qdrant database from.
    port : Optional[int], default=6333
        The port to use when connecting to the Qdrant instance.
    collection_name : Optional[str], default="spikeinterface_documentation"
        The name of the Qdrant collection to create and use.

    Returns
    -------
    Qdrant
        An initialized Qdrant object connected to the specified collection.
    """
    from langchain.vectorstores import Qdrant

    api_key = os.environ["QDRANT_API_KEY"]
    port = 6333
    url = "https://443e1d88-95ec-48b6-b66d-b5c4713ede35.us-east-1-0.aws.cloud.qdrant.io:6333"

    texts = [d.page_content for d in documents]
    metadatas = [d.metadata for d in documents]

    embeddings = OpenAIEmbeddings()
    collection_name = "spikeinterface_documentation"
    qdrant = Qdrant.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        url=url,
        api_key=api_key,
        port=port,
        collection_name=collection_name,
    )

    return qdrant


def generate_croma_database_from_docs(documents: List[Document], persist_directory: str) -> None:
    """
    Auxiliary function to generate the croma database (vectorstore).

    Parameters
    ----------
    read_the_docs_path : str
        The paths with the read the docs documentation in html
    persist_directory : str
        The directory where the vectorstore database is located.

    """
    from langchain.vectorstores import Chroma

    embedings = OpenAIEmbeddings()
    chroma_vectorstore = Chroma.from_documents(documents, embedding=embedings, persist_directory=persist_directory)
    chroma_vectorstore.persist()
    chroma_vectorstore = None  # This triggers persistency


def generate_documents_from_local_read_the_docs(read_the_docs_path: str) -> Document:
    """
    Generate a list of Document objects from a local Read the Docs directory containing HTML files.

    Parameters
    ----------
    read_the_docs_path : str
        The path to the Read the Docs documentation directory containing HTML files.

    Returns
    -------
    Document
        A list of Document objects generated from the local Read the Docs directory.
    """

    read_the_docs_path = Path(read_the_docs_path)
    assert read_the_docs_path.exists() and read_the_docs_path.is_dir()

    loader = DirectoryLoader(read_the_docs_path, glob="**/*.html")
    loaded_documents = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(
        documents=loaded_documents
    )

    return documents
