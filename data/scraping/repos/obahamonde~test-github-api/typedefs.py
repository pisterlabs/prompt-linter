import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import *
from os import environ
from typing import *
from typing import Dict, List

import pinecone as pc
from aiofauna import *
from dotenv import load_dotenv
from langchain.agents import (AgentExecutor, AgentType, Tool, initialize_agent,
                              tool)
from langchain.chains import (LLMChain, SQLDatabaseChain,
                              create_tagging_chain_pydantic)
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader, JSONLoader, PDFMinerLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import *
from langchain.prompts import (ChatMessagePromptTemplate, ChatPromptTemplate,
                               HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)
from langchain.schema import *
from langchain.tools import *
from langchain.vectorstores import Pinecone
from openai import ChatCompletion

load_dotenv()

ai = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-16k-0613", max_tokens=512)  # type: ignore

T = TypeVar("T", bound="Chainable")

F = TypeVar("F", bound="FunctionModel")

# python to json schema types

M = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# type aliases

Vector = List[float]
MetaData = Dict[str, str]
Template = Union[ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatMessagePromptTemplate]
Message = Union[ChatMessage, HumanMessage, SystemMessage, ChatMessage]
Loeader = Union[PDFMinerLoader, CSVLoader, JSONLoader]


class Chainable(BaseModel):
    """Schema that can be chained with other schemas"""

    _ask_for: List[str] = []
    _answers: List[str] = []

    @classmethod
    def __init_subclass__(cls: Type[T], **kwargs):
        super().__init_subclass__(**kwargs)
        for field in cls.__fields__.values():
            field.required = False
        cls.chain = create_tagging_chain_pydantic(cls, ai)

    @classmethod
    def run(cls: Type[T], text: str) -> T:
        return cls.chain.run(text)  # type: ignore

    @classmethod
    def check_what_is_empty(cls: Type[T], text: str):
        instance = cls.run(text)
        for field in cls.__fields__.keys():
            if getattr(instance, field) is None:
                cls._ask_for.append(field)
        return cls._ask_for

    @classmethod
    def prompt(cls: Type[T], ask_for: List[str]) -> str:
        first_prompt = ChatPromptTemplate.from_template(
            """
            System Message:
            I must ask the user for the following information:
            {ask_for}
            AI Message:  
            """
        )
        info_gathering_chain = LLMChain(llm=ai, prompt=first_prompt)
        ai_chat = info_gathering_chain.run(ask_for=ask_for)
        return ai_chat

    @classmethod
    async def run_until_complete(cls: Type[T], websocket: WebSocketResponse) -> T:
        """
        Communicates with the user via websocket to complete the
        Schema.
        """
        instance = cls()
        fields = instance.__fields__.keys()
        cls._ask_for = [
            field
            for field in fields
            if field not in ["ref", "ts"] and getattr(instance, field) is None
        ]
        while cls._ask_for:
            prompt = cls.prompt(cls._ask_for)
            await websocket.send_str(prompt)
            answer = await websocket.receive_str()
            cls._answers.append(answer)
            instance = cls.run("\n".join(cls._answers))
            cls._ask_for = [
                field
                for field in fields
                if field not in ["ref", "ts"] and getattr(instance, field) is None
            ]
        await websocket.send_str(
            "Thanks so much for your time, Don't hesitate to contact me if you have any questions."
        )
        return instance

class FunctionModel(BaseModel):
    @classmethod
    def openai_schema(cls):
        schema = cls.schema()
        return {
            "name": cls.__name__.lower()+"s",
            "type": "object",
            "description": cls.__doc__,
            "properties": {k: {"type": M[type(v)]} for k, v in schema["properties"].items()},
        }
        
    
class ChatBotModel(FunctionModel):
    """Creates a new Configuration for a Chatbot"""
    chatbot_name: str = Field(default="Bot")
    role: str = Field(default="assistant")
    action: str = Field(default="answer any question")
    topic: str = Field(default="what user is asking")
    goal: str = Field(default="help the user to have an amazing experience")
    personality: str = Field(default="helpful, truthful, creative")
    attitude: str = Field(default="polite") 
    

class DocumentModel(FunctionModel):
    """Inserts Documents into Knowledge Base"""
    namespace: str = Field(default="default")
    text: str = Field(...)



def function_calling(text:str):
    ai = ChatCompletion  # type: ignore
    response = ai.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {
                "role": "system",
                "message": "You are a function orchestrator, you will keep prompting the user until one of the functions signatures is fulfilled and call it"""
            },
            {
                "role":"user",
                "message": text
            },
            ChatBotModel.openai_schema,
            DocumentModel.openai_schema
        ]
    )
    return response


class AgentCity:
    """Agent factory class that creates agents with the given tools and agent type."""
    tools:Dict[str,Tool] = {}
    agents:Dict[str,AgentExecutor] = {}
    
    def __init__(self, tools:Dict[str,Tool]={}, agents:Dict[str,AgentExecutor]={}):
        self.tools = tools 
        self.agents = agents 
    
    def agent_tool(self, name:str, description:str):
        """Decorators that register a function as an agent tool.
        You can also register it on the tools array for the initialize agents factory.
        """
        def decorator(func):
            tool_ = Tool(name=name, description=description, func=func)
            self.tools[name] = tool_
            return tool_
        return decorator

    def create_agent(self,name:str,tools:List[Tool], agent:AgentType=AgentType.OPENAI_FUNCTIONS):
        """Creates an agent executor with the given tools and agent type."""
        agent_executor = initialize_agent(tools=tools,llm=ChatOpenAI(client=None,model="gpt-3.5-turbo-16k-0613",max_retries=6,temperature=0), agent=agent,verbose=True)
        if name in self.agents:
            raise Exception("Agent already exists")
        self.agents[name] = agent_executor
        return agent
    
    async def run(self, text:str, name:str):
        """Runs the agent with the given text and agent name."""
        if name not in self.agents:
            raise Exception("Agent not found")
        agent_executor = self.agents[name]
        try:    
            return await agent_executor.arun(input=text)
        except NotImplementedError:
            return agent_executor.run(text=text)