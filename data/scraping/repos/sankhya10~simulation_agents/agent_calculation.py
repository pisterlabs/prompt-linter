import logging
import math
import faiss
import yaml
import json
import pickle

from datetime import datetime, timedelta
from typing import List
from termcolor import colored
#from dotenv import load_dotenv
#from simulate_agent import LLM

from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.experimental.generative_agents import (
    GenerativeAgent,
    GenerativeAgentMemory,
)

def create_llm(api_key_input):
    global LLM
    LLM = ChatOpenAI(max_tokens=1500,model_name='gpt-3.5-turbo',openai_api_key=api_key_input)

def relevance_score_fn(score: float) -> float:
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever(api_key):
    embeddings_model = OpenAIEmbeddings(openai_api_key=api_key)
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, 
        other_score_keys=["importance"], 
        k=15
    )


def create_initial_summary(user_agents,api_key):


    initial_summary= {}    
    agents = {}
    agents_memory = {}
    agent_names = []

    for each_agent in user_agents:

        agent_names.append(each_agent["name"][0])

        agents_memory[each_agent["name"][0]] = GenerativeAgentMemory(
                llm=LLM,
                memory_retriever=create_new_memory_retriever(api_key),
                verbose=False,
                reflection_threshold=5,
            )
        agents[each_agent["name"][0]] = GenerativeAgent(
                name=each_agent["name"][0],
                age=each_agent["age"][0],
                traits=each_agent["Traits"][0] + ", " + each_agent["gender"][0],  
                status=each_agent["status"][0], 
                memory_retriever=create_new_memory_retriever(api_key),
                llm=LLM,
                memory=agents_memory[each_agent["name"][0]],
            )
        for memory in each_agent["initial_memories"]:
                agents[each_agent["name"][0]].memory.add_memory(memory)

    for name in agent_names:
        initial_summary[name] = agents[name].get_summary(force_refresh=True)
    
    

    return agents, initial_summary



