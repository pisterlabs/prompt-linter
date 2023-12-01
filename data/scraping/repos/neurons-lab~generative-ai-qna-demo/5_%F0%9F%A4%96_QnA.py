#!/usr/bin/env python
"""QnA
Version: 0.3.0"""
import os
import streamlit as st
import boto3
from langchain.chat_models import ChatOpenAI
from ai_predict_through_doc import AIPredictThroughDoc



st.set_page_config(
    page_title="Neurons Lab Demo Website",
    # page_icon=":robot:",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MainConstructor class
class MainConstructor:
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
    def __init__(self):
        self.user_input = ''
        if 'qna2' not in st.session_state:
            st.session_state['qna2'] = {}
        self.session_state = st.session_state.qna2
        if "chat_history" not in self.session_state:
            self.session_state["chat_history"] = []
        if 'ai_predict' not in self.session_state:
            self.session_state['ai_predict'] = AIPredictThroughDoc()
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
            if not self.session_state["chat_history"]:
                response = self.session_state['ai_predict'].predict(self.user_input)
                self.session_state["chat_history"] = [(self.user_input, response["answer"])]
                self.session_state['generated'].append(response["answer"])
            else:
                print(self.user_input, self.session_state["chat_history"])
                response = self.session_state['ai_predict'].predict(self.user_input, self.session_state["chat_history"])
                self.session_state["chat_history"].append((self.user_input, response["answer"]))
                self.session_state['generated'].append(response["answer"])
        return response
    
    # render the chat history
    def chat_history_render(self):
        '''Render the chat history'''
        if self.session_state['generated'] and self.session_state['past']:
            st.write('**Chat History:**')
            for i in range(len(self.session_state['generated'])-1, -1, -1):
                st.markdown(f'**Bot:** {self.session_state["generated"][i]}')
                st.markdown(f'**You:** {self.session_state["past"][i]}')
    
    # render the chat widget
    def render(self):
        '''Render the chat widget'''
        self.get_user_input()
        self.predict()
        self.chat_history_render()


# Widget
page_widget = MainConstructor(
    title='QnA Adviser :brain:',
    image='img/NeuronsLab.png'
)
page_widget.render()

# Chat widget
chatbot = Chat()
chatbot.render()


if __name__ == '__main__':
    pass
