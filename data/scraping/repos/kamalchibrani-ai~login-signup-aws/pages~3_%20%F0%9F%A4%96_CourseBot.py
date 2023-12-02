'''
1 the first query will always be from the prompt
2 next query should user input and the output should be based on previous query answer

to achieve this we need to store and pass previous query answer.

'''

import streamlit as st
from streamlit_chat import message
import openai
import os
from dotenv import load_dotenv
from utils import logout_button_sidebar,switch_page_if_auth_isFalse,EmailUs
from streamlit_extras.switch_page_button import switch_page
import time as T
import datetime
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

logout_button_sidebar()
switch_page_if_auth_isFalse()
EmailUs()

print(st.session_state.result)
if st.session_state.result == None:
    switch_page('Profile')

try:
    if st.session_state.query is not None:
        prompt = [
                {
                    'role': 'assistant','content': 'I am an academic consultant and i will do the following and only provide crisp information about the asked query and take content into context'
                },
                {
                    "role": "user","content": f'{st.session_state.query}'
                },
        ]
        st.session_state['message_history'] = prompt
        print(st.session_state.message_history)
        with st.spinner('generating...'):
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=prompt,
                temperature=0.5,
            )
        st.session_state.message_history.append({"role": "assistant", "content": f"{completion.choices[0].message['content']}"})
        st.session_state.query = None
except Exception as e:
    switch_page('Profile')
    print(e)
    

message_history = st.session_state.message_history
print(message_history)
user_input = st.text_input('please insert a question')
user_input = user_input.lstrip()
print(user_input)
if len(user_input)>0:
    print('inside user input',message_history)
    message_history.append({"role": "user", "content": f"{user_input}"})
    with st.spinner('generating...'):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message_history,
            temperature=0.7,
        )
    last_generated_content = completion.choices[0].message['content']
    message_history.append({"role": "assistant", "content": f"{last_generated_content}"})

print('message history after user input',message_history)


if len(message_history)>0:
    for i in range(len(message_history)-1, 1, -2):
        message(message_history[i]['content'],key=str(i))
        message(message_history[i-1]['content'],is_user=True, key=str(i-1))

    save_chat = st.download_button('save_chat',str(message_history),file_name=f'{st.session_state.username}_{datetime.datetime.now().date()}_chat_history.txt')

