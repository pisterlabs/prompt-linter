import streamlit as st

from langchain.schema import AIMessage, SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI

from config import *
import os

os.environ['OPENAI_API_KEY'] = openai
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)


st.title("AssistantX")

if "axmessages" not in st.session_state:
    st.session_state["axmessages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.axmessages:
    st.chat_message(msg["role"]).write(msg["content"])
if message:= st.chat_input(placeholder="Type a message..."):
    st.session_state.axmessages.append({"role": "user", "content": message})
    st.chat_message("user").write(message)
    # update_chat()
    messages = [SystemMessage(content=f"""Hello! I am AssistantX, your AI personal assistant. With {Years_of_Experience} years of experience in managing busy schedules and providing personalized support, I am here to make your life easier. Leveraging my {Industry_Experience} background and knowledge gained from working at companies like {Past_Companies}, I can assist you in organizing your calendar, setting reminders, handling travel arrangements, and suggesting personalized recommendations as well. Just let me know what you need assistance with, and I'll take care of it for you!"""),
            HumanMessage(content="You: " + str(message))]
    response = chat(messages)
    st.session_state.axmessages.append({"role": "assistant", "content": response.content})
    
    st.chat_message("assistant").write(response.content)
    # update_chat()
     
    
    
