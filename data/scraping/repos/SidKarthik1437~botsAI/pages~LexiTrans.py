import streamlit as st

from langchain.schema import AIMessage, SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI

from config import *
import os

os.environ['OPENAI_API_KEY'] = openai
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)


st.title("LexiTrans")

if "ltmessages" not in st.session_state:
    st.session_state["ltmessages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.ltmessages:
    st.chat_message(msg["role"]).write(msg["content"])
if message:= st.chat_input(placeholder="Type a message..."):
    st.session_state.ltmessages.append({"role": "user", "content": message})
    st.chat_message("user").write(message)
    # update_chat()
    messages = [SystemMessage(content=f"""Hello, I am LexiTrans, your AI legal translator. With fluency in multiple languages and a strong understanding of legal terminology, I specialize in accurate and reliable translations of legal documents. Leveraging my expertise in {Industry_Experience} and {Years_of_Experience} years of working with clients like {Past_Companies}, I ensure that the translated content maintains its legal integrity and is culturally appropriate. Whether you need to translate contracts, court documents, patents, or legal agreements, I can provide precise translations that meet your specific requirements. How can I assist you with your legal translation needs today?"""),
                HumanMessage(content="You: "+message)]
    response = chat(messages)
    st.session_state.ltmessages.append({"role": "assistant", "content": response.content})
    
    st.chat_message("assistant").write(response.content)
    # update_chat()
     
    
    
