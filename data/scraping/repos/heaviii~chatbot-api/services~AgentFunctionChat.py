import json
import os
import threading
from langchain.vectorstores import VectorStore, Chroma
from langchain.vectorstores.redis import Redis
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, CombinedMemory, ConversationSummaryMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory

import pinecone 
import getpass


from llama_index import SimpleDirectoryReader, LLMPredictor, ServiceContext
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQA
from loguru import logger
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI,VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)
from config import Config
from services.ThreadedGenerator import ChainStreamHandler, ThreadedGenerator
from tools.knowboxOriginTool import KnowboxOriginTool
from tools.knowboxTool import KnowboxTool
from vectordb.ChromaDb import ChromaDb
from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain

os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
class AgentFunctionChat:

    def __init__(self) -> None:
        logger.info("init")

    def askQuestionChroma(self, userId, prompt):
        session_id = "session_id:"+userId
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

        llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
       
        tools = [
            KnowboxTool(),
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="useful for when you need to answer questions about math"
            ),
        ]
        llm = ChatOpenAI(temperature=0, verbose=True, model="gpt-3.5-turbo-0613")
        memory = self.get_memory(session_id)
        template = self.get_template()
        #create_vectorstore_agent(tools, llm, memory=memory, template=template, verbose=True)
        
        #agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory, template=template,handle_parsing_errors=True)
        #agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory, template=template,handle_parsing_errors=True)
        #agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
        agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
        res = agent.run({"input": prompt})
        print("res:---x",res)

        return res
    
    def askQuestionChromaStream(self, userId, prompt):
        generator = ThreadedGenerator()
        threading.Thread(target=AgentFunctionChat.askQuestionChromaStreamDo, args=(generator, userId, prompt)).start()
        return generator
    
    #还没调通
    def askQuestionChromaStreamDo(generator, userId, prompt):
        try:
            session_id = "session_id:"+userId
            tools = [
                KnowboxOriginTool(),
            ]
            llm = ChatOpenAI(temperature=0, streaming=True, callback_manager=CallbackManager([ChainStreamHandler(generator),StreamingStdOutCallbackHandler]), verbose=True)
            memory = AgentFunctionChat().get_memory(session_id)
            template = AgentFunctionChat().get_template()

            #agent = create_vectorstore_agent(tools=tools, llm=llm, memory=memory, template=template, verbose=True)
        
            agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory, template=template, handle_parsing_errors=True)
            result = agent.run(prompt)
            # 进行问答
            
            print("result:---",result)
            return result

        finally:
            generator.close()
    
    def create_index_chromadb(self):
        return KnowboxTool().init_tool_db()
    
    def get_memory_db(self, session_id):
        message_history = RedisChatMessageHistory(url=Config.REDIS_URL, ttl=Config.REDIS_TTL, session_id=session_id)
        #message_history.clear()
        return message_history
    
    def get_memory(self, session_id):
        message_history = self.get_memory_db(session_id)
        #print("message_history:---",(message_history.messages))
        conv_memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="input",
            chat_memory = message_history,
        )
        
        summary_memory = ConversationSummaryMemory(llm=OpenAI(verbose=True), input_key="input", chat_memory = message_history)
        # Combined
        #print("conv_memory:---",summary_memory.chat_memory.messages)
        memory = CombinedMemory(memories=[conv_memory, summary_memory])
        return memory
    
    def get_template(self):
        _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
        
        Summary of conversation:
        {history}
        Current conversation:
        {chat_history_lines}
        Human: {input}
        AI:"""
        PROMPT = PromptTemplate(
            input_variables=["history", "input", "chat_history_lines"], template=_DEFAULT_TEMPLATE
        )
        return PROMPT


