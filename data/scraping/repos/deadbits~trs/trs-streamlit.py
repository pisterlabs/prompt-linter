import os
import sys
import openai
import argparse
import tiktoken
import streamlit as st
import streamlit_mermaid as stmd
import pandas as pd

from streamlit_extras.badges import badge

from datetime import datetime
from loguru import logger

from typing import Tuple

from rich.console import Console
from rich.markdown import Markdown
from colored import Fore, Back, Style

from trs.main import TRS


st.set_page_config(
    page_title='TRS',
    page_icon='🦊',
    layout='wide',
    initial_sidebar_state='expanded'
)


def clear_chat_history():
    st.session_state.messages = []


def main():
    st.header('TRS - Threat Report Summarizer')
    st.subheader('Web Playground', divider='rainbow')

    pages = ['Home', 'Chat']

    with st.sidebar:
        st.header('🦊 trs', divider='rainbow')
        st.write('[documentation](https://trs.deadbits.ai) | [github](https://github.com/deadbits/trs)')
        badge(type='github', name='deadbits/trs')
        st.divider()

    page = st.sidebar.radio(
        'Select a page:',
        [
            'Analyze',
            'Chat',
            'Database',
            'History',
        ]
    )

    if page == 'Analyze':
        if 'history' not in st.session_state:
            st.session_state.history = []

        url = st.text_input('Enter URL to process:', key='url_input')
        prompt_dir = 'prompts/'
        prompt_list = [prompt.replace('.txt', '') for prompt in os.listdir(prompt_dir) if prompt.replace('.txt', '') not in ['qna', 'mindmap', 'custom1']]

        prompt_name = st.selectbox('Select a prompt:', prompt_list, key='prompt_select')

        response = None
        iocs = None
        mindmap = None

        if st.button('Submit', key='process_button'):
            if url:
                if prompt_name == 'detect':
                    with st.spinner('Processing...'):
                        response = trs.detections(url=url)
                elif prompt_name == 'summary':
                    with st.spinner('Processing...'):
                        summ, mindmap, iocs = trs.summarize(url=url)
                        response = summ  # or whatever you want to assign to response
                elif prompt_name:
                    with st.spinner('Processing...'):
                        response = trs.custom(url=url, prompt=prompt_name)

            if response is not None:
                st.session_state['history'].append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'url': url,
                    'prompt': prompt_name,
                    'response': response
                })
                st.write(response)

                if iocs:
                    st.subheader('IOCs')
                    st.write(iocs)
                
                if mindmap:
                    st.subheader('Mindmap')
                    stmd.st_mermaid(mindmap)

            else:
                st.write('No response received.')
        else:
            st.warning('Please enter a URL to process.')

    elif page == 'Chat':
        # Initialize chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

        chat_input = st.text_input('Message:', key='chat_input')

        if chat_input:
            # Add user message to chat history
            st.session_state.messages.append({'role': 'user', 'content': chat_input})

            # Display user message in chat message container
            with st.chat_message('user'):
                st.markdown(chat_input)

            with st.spinner('Thinking...'):
                qna_answer = trs.qna(prompt=chat_input)

                # Add assistant response to chat history
                st.session_state.messages.append({'role': 'assistant', 'content': qna_answer})

            # Display assistant response in chat message container
            with st.chat_message('assistant'):
                st.markdown(qna_answer)


    elif page == 'History':
        st.title('History')
        # Sort history by timestamp (newest first)
        sorted_history = sorted(
            st.session_state['history'], key=lambda x: x['timestamp'], reverse=True
        )

        for item in sorted_history:
            st.write('Timestamp:', item['timestamp'])
            st.write('URL:', item['url'])
            st.write('Prompt:', item['prompt'])
            st.write('Response:', item['response'])
            st.write('-' * 50)
    
    elif page == 'Database':
        st.title('Database Viewer')
        st.markdown(f'**Total records:** {trs.vdb.count()}')
        st.caption('Viewer restricted to 10k records')
        data = trs.vdb.get()
        df = pd.DataFrame.from_dict(data)
        df = df.head(10000)
        st.dataframe(df.iloc[:,[0, 3, 2, 1]])


if __name__ == '__main__':
    trs = TRS(openai_key=st.secrets['OPENAI_API_KEY'])
    main()