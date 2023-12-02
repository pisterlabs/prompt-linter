# Import Libraries
import os
from dotenv import load_dotenv

from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
from genai.credentials import Credentials

load_dotenv()

# Get the API key and URL from the environment variables
api_key = os.getenv("GENAI_KEY", None)
api_url = os.getenv("GENAI_API", None)

# Create a Credentials object to pass to the LangChainInterface
creds = Credentials(api_key, api_endpoint=api_url)

# Create a GenerateParams object to pass to the LangChainInterface
params = GenerateParams(
            decoding_method="greedy",
            max_new_tokens=1000,
            min_new_tokens=200,
            temperature=0.7,
        )

# Create a LangChainInterface object to use for generating text
llm = LangChainInterface(model="meta-llama/llama-2-70b-chat", params=params, credentials=creds)

# Define the generate_text function
def generate_text(prompt: str) -> str:
    """
    Generates text based on a given prompt using the LangChainInterface.

    Args:
        prompt (str): The prompt to generate text from.

    Returns:
        str: The generated text.
    """
    # Call the generate_text method of the LangChainInterface and return the generated text
    return llm.generate_text(prompt)
