import os
import boto3
import streamlit as st
from utils import get_openai_api_key
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class AIPredict:
    '''AI Predict'''
    def __init__(self, system_chat_prompt, first_prompt, openai_api_key):
        self.openai_api_key = openai_api_key
        self.prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_chat_prompt),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ])
        self.chat = ChatOpenAI(temperature=0.9, openai_api_key=self.openai_api_key)
        self.memory = ConversationBufferMemory(return_messages=True)
        self.conversation = ConversationChain(memory=self.memory, prompt=self.prompt, llm=self.chat, verbose=True)
        self.first_ai_replica = self.predict(input=first_prompt)

    # AI predict function
    def predict(self, input):
        '''AI predict function'''
        ai_replica = self.conversation.predict(
            input=input
        )
        return ai_replica
    
if __name__ == '__main__':
    # AWS Region
    region_name = os.getenv('AWS_REGION', 'us-east-1') 
    # Set SSM Parameter Store name for the OpenAI API key and the OpenAI Model Engine
    API_KEY_PARAMETER_PATH = '/openai/api_key'
    # Create an SSM client using Boto3
    ssm = boto3.client('ssm', region_name=region_name)
    # Get the OpenAI API key from the SSM Parameter Store
    openai_api_key = get_openai_api_key(ssm_client=ssm, parameter_path=API_KEY_PARAMETER_PATH)
    # Create an instance of the AIPredict class
    ai_predict = AIPredict(system_chat_prompt="", first_prompt="", openai_api_key=openai_api_key)
    # Get the first AI replica
    first_ai_replica = ai_predict.first_ai_replica
    # Print the first AI replica
    print(first_ai_replica)
    # print second AI replica
    print(ai_predict.predict(input='my name is Artem. I am feeling very stressed and anxious.'))
    # print third AI replica
    print(ai_predict.predict(input='what is my name?'))
