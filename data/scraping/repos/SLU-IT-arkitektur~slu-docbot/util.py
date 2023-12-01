import tiktoken
import openai
import os
from dotenv import load_dotenv
load_dotenv()

 
openai.api_key = os.environ.get("OPENAI_API_KEY")

def truncate_text(text: str, max_tokens: int) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    truncated_text = encoding.decode(tokens[:max_tokens])
    return truncated_text

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_embedding(text, model="text-embedding-ada-002"): # max tokens 8191
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

