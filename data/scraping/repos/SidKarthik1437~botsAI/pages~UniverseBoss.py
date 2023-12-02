import streamlit as st

from langchain.schema import AIMessage, SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI

from config import *
import os

os.environ['OPENAI_API_KEY'] = openai
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)


st.title("UniverseBoss")

if "ubmessages" not in st.session_state:
    st.session_state["ubmessages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.ubmessages:
    st.chat_message(msg["role"]).write(msg["content"])
if message:= st.chat_input(placeholder="Type a message..."):
    st.session_state.ubmessages.append({"role": "user", "content": message})
    st.chat_message("user").write(message)
    # update_chat()
    messages = [SystemMessage(content=f"""Greetings! I am UniverseBoss, your knowledgeable industrial mentor and consultant. With deep expertise in {Industry_Experience} and a successful track record of helping businesses overcome challenges and achieve their goals, I'm here to guide and support you. Drawing on my experience at companies like {Past_Companies} and {Years_of_Experience} years in the industry, I can provide valuable insights and practical advice tailored to your specific needs. Whether you require assistance with strategic planning, process optimization, cost reduction, or implementing new technologies, let's collaborate to drive your business forward!"""),
                HumanMessage(content="You: "+message)]
    response = chat(messages)
    st.session_state.ubmessages.append({"role": "assistant", "content": response.content})
    
    st.chat_message("assistant").write(response.content)
    # update_chat()
     
    
    
