"""Pinecone Knowledge Base Search

This module facilitates the processing and handling of a document repository by applying vector embedding techniques to PDF files for efficient querying of relevant data.

It consists of the following functionalities:

    read_pdfs: Takes in a path to a folder containing PDF files, reads them and returns a list of documents.
    split_documents: Splits the raw text from the documents into chunks for further processing.
    prereqs_for_embeds: Prepares the prerequisites for the embedding function by extracting the documents' content and metadata.
    pinecone_init: Initializes a Pinecone index which is a vector database for embedding and querying the documents.
    pinecone_insert: Inserts the processed text and metadata into the Pinecone index using an embedding function.
    query_pinecone_index: Queries the Pinecone index to find relevant documents based on user queries.
    pinecone_delete: Deletes a Pinecone index when no longer needed.

The module uses Pinecone for efficient similarity search and retrieval of documents based on user queries. It also uses CohereEmbeddings for creating vector representations of the documents, and PyPDFDirectoryLoader for reading PDF documents from a specified directory.

Please note that parse_user_info is currently not being used and may be integrated in future versions for more personalized responses.
"""
import pinecone
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter


def read_pdfs(
    folder_path: str = "../notebooks/pdfs/",
):
    """
    This function is simply taking the path to the PDF file, loading all the PDFs
    inside the folder and returning them as a List of Documents.

    Params:
        folder_path (str, optional): The path to the folder with PDF files

    Returns:
        documents (List[Documents]): The list of documents
    """
    loader = PyPDFDirectoryLoader(path=folder_path)
    return loader.load()


def split_documents(documents) -> list[str]:
    """
    This function is used to split the raw text into chunks

    Params:
        raw_text (str) : raw text collected from the PDF file

    Returns:
        texts (List[str]): a list of text chunks
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return text_splitter.split_text(documents)


def prereqs_for_embeds(texts):
    docs = []
    metadatas = []
    for text in texts:
        docs.append(text.page_content)
        metadatas.append(text.metadata)

    return {"documents": docs, "metadatas": metadatas}


def pinecone_init(
    indexname: str = "pinecone-knowledge-base", vector_dimension: int = 4096
):
    """
    This function is used to initialize the Pinecone index.

    Params:
        indexname (str, optional): The name of the index (or simople the name of the database we are creating)

    Returns:
        index (Index): client for interacting with a Pinecone index via REST API
    """
    pinecone.init()

    if indexname not in pinecone.list_indexes():
        # we create a new index if not exists
        pinecone.create_index(
            name=indexname,
            metric="cosine",
            dimension=vector_dimension,
        )

    return pinecone.Index(indexname)


def pinecone_insert(
    index: pinecone.Index,
    docs,
    metadatas,
):
    """
    This function is used to insert the documents into the Pinecone index. Please give it some
    time to index the documents before querying it.

    Params:
        index (Pinecone) : Pinecone index object
        docs (list[str]) : a list of text chunks
        metadatas (list[dict['str': 'str']]) : a list of metadata for each text chunk

    Returns:
        None
    """
    embedding_function = CohereEmbeddings()  # type: ignore
    vectorstore = Pinecone(index, embedding_function.embed_query, "text")
    vectorstore.add_texts(docs, metadatas)

    return


def pinecone_delete(indexname: str = "pinecone-knowledge-base"):
    """
    This function is used to delete the Pinecone index given its name
    """
    pinecone.init()

    if indexname in pinecone.list_indexes():
        pinecone.delete_index(indexname)

    return
