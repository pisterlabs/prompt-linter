from llama_index import download_loader, SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = 'OPENAI_API_KEY'

def build_index(url):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 1
    chunk_size_limit = 256

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-4", max_tokens=num_outputs))

    UnstructuredURLLoader = download_loader("UnstructuredURLLoader")
    loader = UnstructuredURLLoader(urls=[url], continue_on_failure=False, headers={"User-Agent": "value"})
    documents = loader.load()
    index = GPTVectorStoreIndex.from_documents(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    return index
