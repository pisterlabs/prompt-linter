import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
from langchain import verbose

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

def init():

    load_dotenv()
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")





    st.set_page_config(page_title="Chatbot Esprit", page_icon=":robot_face:")





def main():
    init()

    chat = ChatOpenAI(temperature=0)

    if "messages" not in st.session_state:

       st.session_state.messages = [
         SystemMessage(content="you are a helpful assistant ")
        ]       







    st.header("Your own chatbot 💬 ")







    with st.sidebar:
        user_input = st.text_input("your message:", key="user_input")
    if user_input:
        #message(user_input, is_user=True)
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Thinking..."):
          response = chat(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))
       # message(response.content, is_user=False)
    
    messages = st.session_state.get("messages", [])
    for i, msg in enumerate(messages[1:]):
      if i % 2 == 0:
        message(msg.content, is_user=True, key=str(i) + '_user')
      else:
        message(msg.content, is_user=False, key=str(i) + '_ai')
       
    
    
if __name__ == '__main__':
    main()

