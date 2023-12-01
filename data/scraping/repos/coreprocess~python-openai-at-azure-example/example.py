# load the .env file
__import__("dotenv").load_dotenv()

# import required libraries
import openai
from os import environ
from tenacity import (retry, retry_if_exception_type, stop_after_attempt, wait_exponential)

# configure the openai library
openai.api_type = "azure"
openai.api_base = environ["AZURE_OPENAI_ENDPOINT"]
openai.api_key = environ["AZURE_OPENAI_KEY"]
openai.api_version = "2023-05-15"

# openai retry utility to avoid rate limits and network errors
@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=2, max=60),
       retry=(retry_if_exception_type(openai.error.RateLimitError)
              | retry_if_exception_type(openai.error.APIConnectionError)
              | retry_if_exception_type(openai.error.Timeout)))
def openai_chat_completion_with_retry(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

# example chat completion
result = openai_chat_completion_with_retry(
    deployment_id=environ["AZURE_OPENAI_DEPLOYMENT_GPT4"],
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": "Hello world",
    }],
    temperature=0,
)

print(result.choices[0].message.content)
