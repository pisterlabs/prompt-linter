import streamlit as st
import random
from PIL import Image
import time
import urllib.request

import cohere
import numpy as np

from Home import nav_page,nameToImg

if "character" not in st.session_state:
    st.session_state['character'] = 'Default'

if "co" not in st.session_state:
    st.session_state['co'] = cohere.Client("cKdIk0HQBozrDy02qQ3VOuglzeK4HzJzdBDHeBZh")

def generate_text(prompt, chat_history, temp=1):
    response = st.session_state['co'].chat(
        prompt,
        model='command',
        temperature=temp,
        chat_history=chat_history,
    )

    # response = st.session_state['co'].generate(  
    #     model='command-nightly',  
    #     prompt = prompt,  
    #     max_tokens=100, # This parameter is optional. 
    #     temperature=0.750,
    #     chat_history=chat_history
    # )

    answer = response.text
    # return response.generations[0].text

    # add message and answer to the chat history
    user_message = {"user_name": "User", "text": prompt}
    bot_message = {"user_name": "Chatbot", "text": answer}
    chat_history.append(user_message)
    chat_history.append(bot_message)

    return answer

message_start = f"I will prompt you with a rap battle stanza, you have to respond according to it acting like {st.session_state['character']}, and referencing moments of its life, you should replay ALWAYS in the format of 4 line stanza, understand?"

# Initialize chat history
if 'chat_history_1' not in st.session_state:
    st.session_state['chat_history_1'] = []
    generate_text(message_start,st.session_state['chat_history_1'])

with st.sidebar:
    
    # urllib.request.urlretrieve( 
    #     'https://cdn6.aptoide.com/imgs/a/d/1/ad139cadd0c58b7a155e60512faa1de0_icon.png', 
    #     'bot.png')
    
    # img = Image.open('bot.png')
    st.image(nameToImg[st.session_state['character']], caption=st.session_state['character'])

st.title("Rap battle against " + st.session_state['character'])



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"],avatar=message["avatar"]):
        st.text(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # print(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar":"human"})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.text(prompt)

    # Display assistant response in chat message container
    with st.chat_message("COnvoker", avatar=nameToImg[st.session_state['character']]):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = generate_text(prompt, st.session_state['chat_history_1'])
        
        # Simulate stream of response with milliseconds delay
        for char in assistant_response:
            full_response += char
            time.sleep(0.01)
                
            # Add a blinking cursor to simulate typing
            message_placeholder.text(full_response + "▌")
        message_placeholder.text(full_response)

        message_placeholder.text(assistant_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response, "avatar":nameToImg[st.session_state['character']]})