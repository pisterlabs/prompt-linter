# streamlit stuff
import streamlit as st
from streamlit_chat import message

# langchain stuff
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import ElasticVectorSearch
from langchain.vectorstores import ElasticsearchStore
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader

# environmental stuff
import os
from dotenv import load_dotenv

# support stuff
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, RequestError, BadRequestError

from icecream import ic

# import logging
# logger = logging.getLogger('my_logger')
# logger.setLevel(logging.DEBUG) # or any level you need

# # create console handler and set level to debug
# handler = logging.StreamHandler()
# handler.setLevel(logging.INFO)

# # create formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# # add formatter to handler
# handler.setFormatter(formatter)

# # add handler to logger
# logger.addHandler(handler)

# all the variables in the .env file that we care about
env_list = ("elasticsearch_user", 
            "elasticsearch_pw", 
            "elasticsearch_host", 
            "elasticsearch_port", 
            "elasticsearch_model_id", 
            "elasticsearch_cloud_id", 
            "elasticsearch_index", 
            "openai_api_key")

def load_env(env_path=None):
    """
    This function loads environment variables from a .env file using the python-dotenv library, 
    and then returns a dictionary-like object containing the environment variables.

    Parameters:
    env_path (str): Optional. The path to the .env file. If not provided, the load_dotenv() function 
                    will look for the .env file in the current directory and if not found, then 
                    recursively up the directories. Defaults to None.

    Returns:
    os._Environ: This is a dictionary-like object which allows accessing environment variables.
    """
    # Load environment variables from a .env file
    load_dotenv(env_path)

    # Return a dictionary-like object containing the environment variables
    return os.environ

def load_session_vars(os_env_vars={}, 
                      key_list=None):
    """
    This function loads variables into Streamlit's session state.

    Parameters:
    os_env_vars (dict): Optional. A dictionary of environment variables to load into the session state. 
                        Defaults to an empty dictionary.
    key_list (list): Optional. A list of keys. If provided, only environment variables 
                     with these keys will be loaded into the session state. Defaults to None.

    The function first checks if key_list is provided. If so, it creates a new dictionary
    with key-value pairs where keys are elements in key_list and the values are corresponding
    values from os.environ.
    
    If key_list is not provided, it uses the os_env_vars dictionary.
    
    It then logs the dictionary of variables that will be loaded into the session state.

    Finally, it iterates over the dictionary of variables, logging each one, and adds each 
    variable to the session state.
    """
    # Check if key_list is provided
    if key_list is not None:
        # If so, create a new dictionary with key-value pairs for each key in key_list
        vars = {env: os.environ[env] for env in key_list if env in os.environ}
    else:
        # If not, use the os_env_vars dictionary
        vars = os_env_vars


    # Iterate over the dictionary of variables
    for key, value in vars.items():
        # Log each variable
        ic(f"Setting st.{key} = {value}")

        # Add the variable to the session state
        st.session_state[key] = value

def build_app_vars():
    """
    This function builds the es_url for use in the Streamlit app and stores it in the session state.

    It uses the Elasticsearch host, port, user, and password from the session state to construct the URL in the format required.

    Note: It's critical to ensure that the necessary variables (elasticsearch_user, elasticsearch_pw, elasticsearch_host, elasticsearch_port) are available in the session state before calling this function.

    """

    # Construct the Elasticsearch URL using the host, port, user, and password
    # The URL is formatted according to the Elasticsearch's standard URL format

    if st.session_state['elasticsearch_user'] and \
        st.session_state['elasticsearch_pw'] and \
        st.session_state['elasticsearch_host'] and \
        st.session_state['elasticsearch_port']:

        st.session_state['es_url'] = f"https://{st.session_state['elasticsearch_user']}:{st.session_state['elasticsearch_pw']}@{st.session_state['elasticsearch_host']}:{st.session_state['elasticsearch_port']}"
        ic(f"es_url = {st.session_state.es_url}")

    else:
        ic("The es_url could not be written because elasticsearch_user, elasticsearch_pw, elasticsearch_host, and elasticsearch_port are not defined in the session state")
    # we set this as an environment variable because it has to be present
    # in so many different places

    if 'open_api_key' not in st.session_state: 
        os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
    else:
        ic("The open_api_key is not defined in the session state.")

def create_new_es_index(index_name=None, es_url=None, delete_index=False):
    """
    Creates a new Elasticsearch index with a predefined mapping. If the index already exists,
    it will be deleted and a new index with the specified name will be created.

    Parameters:
        index_name (str, optional): The name of the index to be created. Defaults to None.
        es_url (str, optional): The URL of the Elasticsearch cluster. Defaults to None.
        delete_index (bool, optional): Should we delete the index if it exists?  Defaults to False.

    Raises:
        NotFoundError: If the index specified in index_name does not exist.
        RequestError: If there's an error in creating the index.

    Example:
        create_new_es_index(index_name="my_index", es_url="http://localhost:9200")
    """

    # Define the mapping for the index, specifying the data types and structure
    table_mapping = {
        "mappings": {
            "properties": {
                "metadata": {
                    "properties": {
                        "page": {"type": "long"},
                        "row": {"type": "long"},
                        "source": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        }
                    }
                },
                "text": {
                    "type": "text"
                },
                "vector": {
                    "type": "dense_vector", 
                    "index": True,
                    "similarity": "l2_norm",
                    "dims": 1536
                }
            }
        }
    }

    # Connect to the Elasticsearch cluster using the provided URL
    es = Elasticsearch([es_url])

    if delete_index:
        try:
            response = ic(es.indices.delete(index=index_name))
        except NotFoundError:
            ic(f"Index '{index_name}' not found.")
        except Exception as e:
            ic(f"An error occurred: {e}")

    # Check if the index exists
    try:
        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name, mappings=table_mapping["mappings"])

    except NotFoundError as e:
        # Handle case when the index is not found
        ic(f"An error occurred: {e}")

    except (RequestError, BadRequestError) as e:
        # Handle other request errors such as BadRequestError
        ic(f"An error occurred while creating the index: {e}")