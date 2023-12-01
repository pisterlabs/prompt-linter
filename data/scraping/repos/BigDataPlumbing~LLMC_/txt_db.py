import os
import openai 
import streamlit as st

from streamlit_chat import message

openai.api_key = st.secrets["openai"]

content_count = 0



if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
content_concluded = False
content_prompted = False
        


if prompt := st.chat_input("Query:"):

    model= "gpt-3.5-turbo-0613",
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        result = ""
        content_prompted = True
                                
        for response in openai.ChatCompletion.create(
            model= "gpt-3.5-turbo-0613",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            result += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(result + "▌")
            content_count += 1
        message_placeholder.markdown(result)
    st.session_state.messages.append({"role": "assistant", "content": result})
    
    content_concluded = True
    
    
    
    

if content_concluded and content_prompted and content_count > 1:
    print(f"Responses count: {content_count}")

