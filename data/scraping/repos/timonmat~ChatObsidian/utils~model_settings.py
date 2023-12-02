#model_settings.py
import streamlit as st
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, LLMPredictor, PromptHelper, OpenAIEmbedding, ServiceContext 
from llama_index.logger import LlamaLogger
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from enum import Enum

class sentenceTransformers(Enum):
    OPTION1 = "sentence-transformers/all-MiniLM-L6-v2"    #default
    OPTION2 = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    OPTION3 = "sentence-transformers/all-mpnet-base-v2"

def get_sentence_transformer_dropdown():
    options = [e.value for e in sentenceTransformers]
    selected_option = st.selectbox("Sentence transformer:", options)
    return selected_option

def get_embed_model(provider='Langchain', model_name=sentenceTransformers.OPTION1.value):
    # load in HF embedding model from langchain
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_name)) if provider=='Langchain' else OpenAIEmbedding()
    return embed_model

def get_prompt_helper():
    # define prompt helper
    max_input_size = 4096
    num_output = 2048
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)
    return prompt_helper

def get_llm_predictor():
    # define LLM
    num_output = 2048
    
    #llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=num_output))  
    llm_predictor = LLMPredictor(ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo", max_tokens=num_output))
    return llm_predictor

@st.cache_resource
def get_logger():
    llama_logger = LlamaLogger()
    return llama_logger

def get_service_context(llm_predictor=get_llm_predictor(), 
                        embed_model=get_embed_model(), 
                        prompt_helper=get_prompt_helper(), 
                        chunk_size_limit=512, 
                        llama_logger=get_logger()):
    return ServiceContext.from_defaults(llm_predictor=llm_predictor, 
                        embed_model=embed_model, 
                        prompt_helper=prompt_helper, 
                        chunk_size_limit=chunk_size_limit, 
                        llama_logger=llama_logger)
