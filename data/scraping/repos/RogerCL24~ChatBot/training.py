from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
import os
from langchain import OpenAI
import openai

os.environ['OPENAI_API_KEY'] = "<API-KEY>"

max_input = 4098
tokens = 256
chunk_size = 600
max_chunk_overlap = 0.1

def training(path):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    docs = SimpleDirectoryReader(path).load_data()
    Prompt_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)
    model = LLMPredictor(llm=OpenAI(temperature=0,model_name="text-ada-001",max_tokens=tokens))
    context = ServiceContext.from_defaults(llm_predictor=model, prompt_helper=Prompt_helper)

    index_model = GPTVectorStoreIndex.from_documents(docs, service_context=context)
    index_model.storage_context.persist(persist_dir='Store')

training("data")