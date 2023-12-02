from typing import List
from langchain.docstore.document import Document
import pinecone
from healthfirstai_prototype.models.data_models import User
from langchain.vectorstores import Pinecone
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from .chains import load_chain
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or ""
PINECONE_ENV_NAME = os.getenv("PINECONE_ENV_NAME") or ""


def query_pinecone_index(
    query: str,
    indexname: str = "pinecone-knowledge-base",
) -> List[Document]:
    """
    This function is used to query the Pinecone index

    Params:
        query (str) : The user's query / question
        indexname (str) : Name of the Pinecone index object

    Returns:
        response (list[Document]): The response object from the Pinecone index
    """
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV_NAME)
    embedding_function = CohereEmbeddings(client=None)
    docsearch = Pinecone.from_existing_index(indexname, embedding_function)
    return docsearch.similarity_search(query, k=1)


def query_based_similarity_search(
    query: str,
    chain: BaseCombineDocumentsChain,
) -> str:
    """
    This function is used to search through the knowledge base (aka book stored in the PDF file under the notebooks/pdfs/ folder)

    Params:
        query (str): The user's query / question
        chain (BaseCombineDocumentsChain) : The LLM chain object

    Returns:
        The response from the LLM chain object
    """
    docs = query_pinecone_index(query)
    return chain.run(input_documents=docs, question=query)


# FIX: Knowledge base search is currently not working
# It's sending back too many search results for one query
def knowledge_base_search(query: str) -> str:
    """
    This function is used to load the chain and sets it up for the agent to use

    Params:
        query (str) : The user's query / question

    Returns:
        The response from the LLM chain object
    """
    chain = load_chain()
    return query_based_similarity_search(query, chain)


def search_internet(query: str) -> str:
    """
    This function is used to search through the internet (SerpAPI)
    for nutrition/exercise information in case it doesn't require further clarification,
    but a simple univocal answer.

    Params:
        query (str) : The user's query / question

    Returns:
        The response from the SerpAPI's query to Google
    """
    search = GoogleSerperAPIWrapper()
    return search.run(query)


# NOTE: This function is not yet finished
def parse_user_info(user_data: User) -> dict[str, str]:
    """
    This function is used to parse the user's personal information

    Params:
        user_data (int) : User's personal information

    Returns:
        a dictionary containing the user's personal information
    """
    return {
        "height": str(user_data.height),
        "weight": str(user_data.weight),
        "gender": str(user_data.gender),
        "age": str(user_data.dob),
        "city_id": str(user_data.city_id),
        "country_id": str(user_data.country_id),
    }
