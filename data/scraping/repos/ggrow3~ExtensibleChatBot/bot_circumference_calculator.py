import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from chatbot_settings import ChatBotSettings
from typing import Callable, Dict, List, Optional, Union
from langchain.agents import load_tools, initialize_agent
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory,
                                                  ConversationSummaryMemory,
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI
from chatbot_settings import ChatBotSettings
import pinecone
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.agents import initialize_agent
from tool_circumference import CircumferenceTool
from bot_abstract_class import BotAbstract



class BotCirucumferenceTool(BotAbstract):
    def __init__(self, chatBotSettings: ChatBotSettings()):
        self.chatbotSettings = chatBotSettings

        self.llm = chatBotSettings.llm
        self.memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=5,
            return_messages=True
        )

    def get_bot_response(self, text: str):
        i = ChatBotSettings()
      
        # initialize conversational memory
        

        tools = [CircumferenceTool()]

        # initialize agent with tools
        agent = initialize_agent(
            agent='chat-conversational-react-description',
            tools=tools,
            llm=self.llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=self.memory
        )

        print(agent.agent.llm_chain.prompt.messages[0].prompt.template)

        sys_msg = """Assistant is a large language model trained by OpenAI.

        Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

        Unfortunately, Assistant is terrible at maths. When provided with math questions, no matter how simple, assistant always refers to it's trusty tools and absolutely does NOT try to answer math questions by itself

        Overall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
        """
        new_prompt = agent.agent.create_prompt(
            system_message=sys_msg,
            tools=tools
        )

        agent.agent.llm_chain.prompt = new_prompt
            
        response = agent(text)

        print(response)


