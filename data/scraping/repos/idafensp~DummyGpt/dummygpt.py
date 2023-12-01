import openai
import streamlit as st
import time
import random
import numpy as np
import pandas as pd


from  chat_message import ChatMessage

st.title("Dummy GPT  Messages")


if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    msg.display()

if prompt := st.chat_input("What is up?"):

    user_message = ChatMessage("user")
    user_message.add_content("markdown", prompt)
    st.session_state.messages.append(user_message)

    user_message.display()


    # sleep a random amount of time to simulate a long-running process
    sleep_time = random.uniform(0.1, 1)
    time.sleep(sleep_time)


    assistant_message = ChatMessage("assistant")    

    if "graf" in prompt:
        assistant_message.add_content("markdown", "Aqui tienes un grafico")
        assistant_message.add_content("bar_chart", np.random.randn(30, 3))
    if "mapa" in prompt:
        df = pd.DataFrame(
        np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        columns=['lat', 'lon'])
        assistant_message.add_content("map", df)
    if "image" in prompt:
        #image = Image.open('resources/800px-A-Cat.jpg')
        #assistant_message.add_content("image", image)
        pass

    assistant_message.add_content("markdown", "Espero que te guste...")

    st.session_state.messages.append(assistant_message)
    assistant_message.display()

    #st.session_state.messages.append({"role": "assistant", "content": full_response})