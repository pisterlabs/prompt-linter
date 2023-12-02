import os
from dotenv import load_dotenv

from langchain.llms import OpenAI
from langchain.llms import HuggingFaceHub

# Use the environment variables to retrieve API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_KEY = os.getenv("HUGGINGFACEHUB_API_KEY")

# Initialize OpenAI LLM with the API key
llm_openai = OpenAI(model_name="text-davinci-003", api_key=OPENAI_API_KEY)

# Initialize Hugging Face LLM without the api_key parameter
llm_huggingface = HuggingFaceHub(repo_id="google/flan-t5-large")

our_query = "What is python ?"
completion_openai = llm_openai(our_query)
print(completion_openai)

our_query = "Describe about mahendra singh dhoni?"
completion_huggingface = llm_huggingface(our_query, api_key=HUGGINGFACEHUB_API_KEY)
print(completion_huggingface)
