
from pydantic import BaseSettings
class Settings(BaseSettings):
    AOAI_API_KEY :str
    AOAI_ENDPOINT :str
    AOAI_MODEL :str

settings = Settings(_env_file=".env")

from logger import logger

from openai import AzureOpenAI

def call_chatgpt (messages,temperature=0):

    client = AzureOpenAI(
        azure_endpoint = settings.AOAI_ENDPOINT, 
        api_key=settings.AOAI_API_KEY,  
        api_version="2023-10-01-preview",
        timeout=20,
    )

    stream = client.chat.completions.create(
        model=settings.AOAI_MODEL,
        messages=messages,
        temperature=temperature,
        stream = True,
    )

    return stream