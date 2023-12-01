from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import sys
import os

os.environ['OPENAI_API_KEY'] = "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"

def createVectorIndex(path):
    max_input = 4096
    tockens = 256
    chunk_size = 600
    max_chunk_overlap = 20
    
    prompt_helper = PromptHelper(max_input, tockens, chunk_overlap_ratio = 0.1, chunk_size_limit = chunk_size)
    
    llmpredictor = LLMPredictor(llm = ChatOpenAI(temperature = 0, model_name = "text-ada-001", max_tokens = tockens))
    
    docs = SimpleDirectoryReader(path).load_data()
    
    vectorIndex = GPTVectorStoreIndex(docs, llm_predictor = llmpredictor, prompt_helper = prompt_helper)
    
    vectorIndex.save_to_disk('vectorindex.json')
    
    return vectorIndex


createVectorIndex('C:/Users/Administrator/Desktop/Data_Science')
