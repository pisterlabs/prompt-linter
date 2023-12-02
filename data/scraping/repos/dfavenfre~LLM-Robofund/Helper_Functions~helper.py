from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone
from langchain import OpenAI
from dotenv import load_dotenv
import pinecone
import requests
import os

load_dotenv()


def scrape_linkedin_profile(linkedin_profile_url: str):
    """
    Scrapes data from a LinkedIn profile using the Proxycurl API.

    Args:
        linkedin_profile_url (str): The URL of the LinkedIn profile to scrape.

    Returns:
        dict: A dictionary containing scraped data from the LinkedIn profile.

    Note:
        This function sends a request to the Proxycurl API to retrieve information from a LinkedIn profile and cleans the data for relevant information.
    """
    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    header_dic = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}

    response = requests.get(
        api_endpoint, params={"url": linkedin_profile_url}, headers=header_dic
    )

    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None) and k not in ["people_also_viewed"]
    }

    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data


def create_vectordb(index_name: str, metric: str, dimension: int):
    """
    Creates a vectorized database in an existing index on Pinecone, or creates a new one if it doesn't exist.

    Args:
        index_name (str): The name of the Pinecone index to create or use.
        metric (str): The metric to be used for similarity computation (e.g., 'cosine', 'euclidean').
        dimension (int): The number of dimensions for the vectors in the index.

    Raises:
        PineconeException: If there is an issue with the Pinecone API or Environment Value.

    Note:
        This function initializes the Pinecone environment, creates an index if it doesn't exist, and configures the index with the specified metric and dimension.
    """
    pinecone.init(
        api_key=os.environ.get("pinecone_api_key"),
        environment=os.environ.get("pinecone_environment_value"),
    )

    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)

    pinecone.create_index(name=index_name, metric=metric, dimension=dimension)


def chunk_up_documents(
    file_path: str, chunk_size: int, chunk_overlap: int, index_name: str
):
    """
    Process and chunk up documents from a file, creating a vectorized database in a Pinecone index.

    Args:
        file_path (str): The path to the file containing text documents to be indexed.
        chunk_size (int): The size of text chunks for processing.
        chunk_overlap (int): The amount of overlap between text chunks.
        index_name (str): The name of the Pinecone index to create or use.

    Note:
        This function processes the file, splits it into text chunks, and indexes them in a Pinecone database.
    """
    loader = PyPDFLoader(file_path=file_path)
    document = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="\n"
    )
    texts = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)

