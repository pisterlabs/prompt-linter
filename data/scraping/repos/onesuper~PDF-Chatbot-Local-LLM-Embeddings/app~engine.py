
from langchain.agents import (
  initialize_agent,
  Tool,
  AgentType
)
from llama_index.callbacks import (
  CallbackManager,
  LlamaDebugHandler
)
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
)

import os
import openai
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def init_llm_from_env(temperature=0.1, max_tokens=1024):
  llm_type = os.getenv("LLM")

  if llm_type == 'openai':
      from langchain.chat_models import ChatOpenAI

      openai.api_key = os.getenv("OPENAI_API_KEY")
      llm = ChatOpenAI(temperature=temperature,
                       model_name="gpt-3.5-turbo", 
                       max_tokens=max_tokens)

  elif llm_type == 'xinference':
      from langchain.llms import Xinference

      llm = Xinference(
          server_url=os.getenv("XINFERENCE_SERVER_ENDPOINT"), 
          model_uid=os.getenv("XINFERENCE_LLM_MODEL_UID")
        )
  else:
      raise ValueError(f"Unknown LLM type {llm_type}")

  return llm

def init_embedding_from_env(temperature=0.1, max_tokens=1024):
  embedding_type = os.getenv("EMBEDDING")

  if embedding_type == 'openai':
      from llama_index.embeddings import OpenAIEmbedding

      openai.api_key = os.getenv("OPENAI_API_KEY")
      embedding = OpenAIEmbedding()

  elif embedding_type == 'xinference':
      from langchain.embeddings import XinferenceEmbeddings
      from llama_index.embeddings import LangchainEmbedding

      embedding = LangchainEmbedding(
         XinferenceEmbeddings(
            server_url=os.getenv("XINFERENCE_SERVER_ENDPOINT"),
            model_uid=os.getenv("XINFERENCE_EMBEDDING_MODEL_UID")
         )
      )
  else:
      raise ValueError(f"Unknown EMBEDDING type {embedding_type}")

  return embedding

def get_service_context(callback_handlers):
    callback_manager = CallbackManager(callback_handlers)
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=512,
        chunk_overlap=128,
        callback_manager=callback_manager,
    )
    return ServiceContext.from_defaults(
        embed_model=init_embedding_from_env(),
        callback_manager=callback_manager,
        llm=init_llm_from_env(),
        chunk_size=512,
        node_parser=node_parser
    )

def get_storage_context():
    return StorageContext.from_defaults()

def get_langchain_agent_from_index(summary_index, vector_index):
    list_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    vector_query_engine = vector_index.as_query_engine(
        similarity_top_k=3
    )        
    tools = [
        Tool(
            name="Summary Tool",
            func=lambda q: str(list_query_engine.query(q)),
            description="useful for when you want to get summarizations",
            return_direct=True,
        ),
        Tool(
            name="Lookup Tool",
            func=lambda q: str(vector_query_engine.query(q)),
            description="useful for when you want to lookup detailed information",
            return_direct=True,
        ),
    ]

    agent_chain = initialize_agent(
        tools, 
        init_llm_from_env(), 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True
    )
    return agent_chain


def get_query_engine_from_index(index):
    return index.as_query_engine(
        similarity_top_k=3
    )

def get_chat_engine_from_index(index):
    return index.as_chat_engine(chat_mode="condense_question", verbose=True)
class ChatEngine:

    def __init__(self, file_path):
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)

        service_context = get_service_context([llama_debug])
        storage_context = get_storage_context()

        documents = SimpleDirectoryReader(input_files=[file_path], filename_as_id=True).load_data()
        logging.info(f"Loaded {len(documents)} documents from {file_path}")

        nodes = service_context.node_parser.get_nodes_from_documents(documents)        
        storage_context.docstore.add_documents(nodes)
        logging.info(f"Adding {len(nodes)} nodes to storage")
    
        self.summary_index = SummaryIndex(nodes, storage_context=storage_context, 
                                          service_context=service_context)
        self.vector_index = VectorStoreIndex(nodes, storage_context=storage_context,
                                             service_context=service_context)  

    # def conversational_chat(self, query, callback_handler):
    #     """
    #     Start a conversational chat with a agent
    #     """
    #     response = self.agent_chain.run(input=query, callbacks=[callback_handler])
    #     return response

    def conversational_chat(self, query, callback_handler):
        """
        Start a conversational chat with a agent
        """
        return get_chat_engine_from_index(self.vector_index).chat(query).response