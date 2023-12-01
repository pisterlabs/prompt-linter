import os
import openai
from dotenv import load_dotenv
from langchain.llms import AzureOpenAI
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a completion
# llm = AzureOpenAI(deployment_name="text-davinci-003")
# joke = llm("Tell me a dad joke")
# print(joke)

# Azure OpenAI model mapping
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)
# OpenAI를 기본으로 사용할때
# embeddings = OpenAIEmbeddings()
text = "Algoritma is a data science school based in Indonesia and Supertype is a data science consultancy with a distributed team of data and analytics engineers."
doc_embeddings = embeddings.embed_documents([text])

# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# print(OPENAI_API_KEY)
print(doc_embeddings)
