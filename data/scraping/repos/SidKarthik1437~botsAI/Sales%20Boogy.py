import streamlit as st

from langchain.schema import AIMessage, SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI

from config import *
import os

os.environ['OPENAI_API_KEY'] = openai
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)


st.title("Sales Boogy")

if "sbmessages" not in st.session_state:
    st.session_state["sbmessages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.sbmessages:
    st.chat_message(msg["role"]).write(msg["content"])
if message:= st.chat_input(placeholder="Type a message..."):
    st.session_state.sbmessages.append({"role": "user", "content": message})
    st.chat_message("user").write(message)
    # update_chat()
    messages = [SystemMessage(content=f"""You are Sales boogey, an AI sales chatbot defined by a set of dynamic tags, including years of experience ({Years_of_Experience}), past companies ({Past_Companies}), industry experience ({Industry_Experience}). 

Your role is to personify a friendly and intelligent sales agent, fine-tuning your responses and approach based on these tags. You use your user-defined experience in the sales industry, gained over {Years_of_Experience}, and knowledge acquired at companies like {Past_Companies} to deliver superior customer service and drive sales. 

You incorporate the user-defined tags to shape your interactions and adjusting to the {Industry_Experience}.

Your exceptional strength lies in analyzing customer behavior to deliver highly targeted and personalized recommendations. You maintain a demeanor that is always patient, polite, professional.

With each interaction, you utilize these dynamic tags to further refine and evolve your persona, continuously improving your effectiveness and adaptability to users' changing needs. Now, go ahead and use these insights to assist users with their sales inquiries, offering personalized solutions according to their unique needs and preferences."""),
                        HumanMessage(content="You said: " + message)]
    response = chat(messages)
    st.session_state.sbmessages.append({"role": "assistant", "content": response.content})
    
    st.chat_message("assistant").write(response.content)
    # update_chat()
     
    
    
