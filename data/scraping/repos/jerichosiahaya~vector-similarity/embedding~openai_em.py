from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_TYPE = os.environ.get('OPENAI_API_TYPE')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_API_BASE = os.environ.get('OPENAI_API_BASE')
OPENAI_API_VERSION = os.environ.get('OPENAI_API_VERSION')
OPENAI_API_DEPLOYMENT_NAME = os.environ.get('OPENAI_API_DEPLOYMENT_NAME')

def embedder(text: str):

    embeddings = OpenAIEmbeddings(
        deployment=OPENAI_API_DEPLOYMENT_NAME,
        openai_api_version=OPENAI_API_VERSION,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        openai_api_type=OPENAI_API_TYPE
        )

    result = embeddings.embed_query(text)

    return result