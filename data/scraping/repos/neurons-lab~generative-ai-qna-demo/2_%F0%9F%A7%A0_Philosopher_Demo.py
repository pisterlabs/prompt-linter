#!/usr/bin/env python
"""Philosopher Demo
Version: 0.2.0"""
import os
import streamlit as st
import boto3
from utils import get_openai_api_key
from ai_predict import AIPredict
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# AWS Region
region_name = os.getenv('AWS_REGION', 'us-east-1') 
# Set SSM Parameter Store name for the OpenAI API key and the OpenAI Model Engine
API_KEY_PARAMETER_PATH = '/openai/api_key'
# Create an SSM client using Boto3
ssm = boto3.client('ssm', region_name=region_name)
# DynamoDB client to store personal information
dynamodb = boto3.resource('dynamodb', region_name=region_name)

# Get the API key from the SSM Parameter Store
openai_api_key = get_openai_api_key(ssm_client=ssm, parameter_path=API_KEY_PARAMETER_PATH)

SYSTEM_MESSAGE_PROMPT_TEMPLATE = """I want you to act as a philosopher. 
                I will provide some topics or questions related to the study of philosophy, 
                and it will be your job to explore these concepts in depth. 
                This could involve conducting research into various philosophical theories, 
                proposing new ideas or finding creative solutions for solving complex problems.
    """
FIRST_MESSAGE_PROMPT_TEMPLATE = """Start with introduction and initiate convesation.
    Find out how can you help the user.
    """

st.set_page_config(
    page_title="Neurons Lab Demo Website",
    # page_icon=":robot:",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Phylosopher class
class PhylosopherDemo:
    def __init__(self, title, image):
        self.title = title
        self.image = image
        self.language_option = 'English'

    def render(self):
        '''Render'''
        col1, col2 = st.columns([1, 2])
        col1.image(self.image, width=250)
        col2.title(self.title)

        st.sidebar.success('Select Any Page to Start')
        st.sidebar.markdown('''
        **Contact Us:**\n
        info@neurons-lab.com\n
        +44 330 027 2146\n
        https://neurons-lab.com/\n
        ''')
        # disclaimer
        st.markdown('**Disclaimer:** This is a demo application to show capabilities of ChatGPT. The advice is not a substitute for professional advice.')            

# Streamlit Chat class
class Chat:
    '''Streamlit Chat'''
    def __init__(self, language_option, openai_api_key):
        self.language_option = language_option
        self.openai_api_key = openai_api_key
        self.user_input = ''
        if 'phylosopher' not in st.session_state:
            st.session_state['phylosopher'] = {}
        self.session_state = st.session_state.phylosopher
        if 'ai_predict' not in self.session_state:
            self.session_state['ai_predict'] = AIPredict(SYSTEM_MESSAGE_PROMPT_TEMPLATE, FIRST_MESSAGE_PROMPT_TEMPLATE, openai_api_key=self.openai_api_key)
        if 'initial' not in self.session_state:
            self.session_state['initial'] = self.session_state['ai_predict'].first_ai_replica
        if 'user_input' not in self.session_state:
            self.session_state['user_input'] = self.user_input
        if 'generated' not in self.session_state:
            self.session_state['generated'] = []
        if 'past' not in self.session_state:
            self.session_state['past'] = []

    def chat_submit(self):
        '''Clean the chat input after submit'''
        self.session_state['user_input'] = st.session_state['chat_widget']
        st.session_state['chat_widget'] = ''

    def get_user_input(self):
        '''Get user input from the chat widget'''
        # st.text_area(
        st.text_input(
            label='You',
            # height=100,
            max_chars=500,
            key='chat_widget',
            on_change=self.chat_submit
        )
        if self.session_state['user_input']:
            self.user_input = self.session_state['user_input']
            self.session_state['past'].append(self.user_input)
        return self.user_input
    
    # predict the next message
    def predict(self):
        '''Predict the next message'''
        response = ''
        if self.user_input:
            response = self.session_state['ai_predict'].predict(input=self.user_input)
            self.session_state['generated'].append(response)
        return response
    
    # render the chat history
    def chat_history_render(self):
        '''Render the chat history'''
        if self.session_state['generated'] and self.session_state['past']:
            st.write('**Chat History:**')
            for i in range(len(self.session_state['generated'])-1, -1, -1):
                st.markdown(f'**Bot:** {self.session_state["generated"][i]}')
                st.markdown(f'**You:** {self.session_state["past"][i]}')
        st.markdown(f'**Bot:** {self.session_state["initial"]}')
    
    # render the chat widget
    def render(self):
        '''Render the chat widget'''
        self.get_user_input()
        self.predict()
        self.chat_history_render()


# Widget
page_widget = PhylosopherDemo(
    title='Phylosopher Adviser :brain:',
    image='img/NeuronsLab.png'
)
page_widget.render()

# Chat widget
chatbot = Chat(
    language_option=page_widget.language_option,
    openai_api_key=openai_api_key
)
chatbot.render()


if __name__ == '__main__':
    pass
