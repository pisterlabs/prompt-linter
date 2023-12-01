import os

import openai
from langchain.llms import AzureOpenAI

os.environ["OPENAI_API_KEY"] = 'YOUR_OPENAI_API_KEY'

openai.api_type = "azure"
openai.api_base = "https://adt-openai.openai.azure.com/"
openai.api_version = "2022-12-01"
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY", '938ce9d50df942d08399ad736863d063')

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://adt-openai.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "938ce9d50df942d08399ad736863d063"

OPENAI_API_KEY="938ce9d50df942d08399ad736863d063"
PINECONE_API_KEY="33e67396-4ede-4259-b084-73f5cd10098d"
PINECONE_API_ENV="us-east4-gcp"

from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper

documents = SimpleDirectoryReader('data').load_data()
index = GPTSimpleVectorIndex.from_documents(documents)

def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    # define LLM
    llm_predictor = LLMPredictor(llm=AzureOpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')

    return index

construct_index('./data')
index.save_to_disk('index.json')
# load from disk
index = GPTSimpleVectorIndex.load_from_disk('index.json')
