import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()


def get_client():
    return AzureOpenAI(
        api_version="2023-07-01-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
