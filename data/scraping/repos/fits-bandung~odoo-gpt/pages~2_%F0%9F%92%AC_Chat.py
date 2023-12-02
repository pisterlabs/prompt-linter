"""
Skrip Python ini mengintegrasikan paket perangkat lunak ERP, Odoo, dengan layanan pesan WhatsApp, 
menggunakan model bahasa generatif OpenAI, GPT-3.5, sebagai asisten AI. 
Dibangun dengan pustaka Streamlit, Langchain, dan llms, skrip ini memungkinkan interaksi 
berbasis percakapan yang kontekstual dan informatif antara sistem Odoo dan pengguna melalui WhatsApp, 
dengan asisten AI memfasilitasi dan memperkaya proses komunikasi.# Author: Avratanu Biswas

# July 2023
#by Andhitiawarman Nugraha
"""

# Import necessary libraries
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from utils.get_credential import get_credentials, is_valid_token
 
import os
import textwrap

import utils.login as login
import utils.sidebar as sidebar
import utils.whatsapp as wa

from utils.database import Message, User, db_sqlalchemy, app

from utils.chat_utils import show_chat_histories
  

def new_chat():
  """
    Clears session state and starts a new chat.
    """
  save = []
  for i in range(0, len(st.session_state['generated'])):
    save.append("User:" + st.session_state["past"][i])
    save.append("Odoo-GPT:" + st.session_state["generated"][i])

  st.session_state["stored_session"].append(save)
  st.session_state["generated"] = []
  st.session_state["past"] = []
  st.session_state["input"] = ""
  # st.session_state.entity_memory.entity_store = {}
  # st.session_state.entity_memory.buffer.clear()

 


def clear_chat_histories(phone_number):
    with st.expander("Clear Chat History", expanded=True):
    
      if st.checkbox(f"**DELETE** from chat history of [{phone_number}]"):
          if st.button(f"⚠️ Yes, I am sure to CLEAR All the chat history of [{phone_number}]"):
              with app.app_context(): 
                  # model.query.filter(model.recipient.in_(selected_number)).all()
                  Message.query.filter(Message.recipient == phone_number).delete()
                  # delete()
                  db_sqlalchemy.session.commit()
              st.success("✅ **'Message'** table has been **DELETED**.")
              st.experimental_rerun()




def run():
    sidebar.run()
                     
    # Initialize session states
    if "generated" not in st.session_state:
      st.session_state["generated"] = []
    if "past" not in st.session_state:
      st.session_state["past"] = []
    if "input" not in st.session_state:
      st.session_state["input"] = ""
    if "stored_session" not in st.session_state:
      st.session_state["stored_session"] = []
    if "download_str" not in st.session_state:
      st.session_state["download_str"] = []
    if "token" not in st.session_state:
      st.session_state["token"] = []
    if "credentials" not in st.session_state:
      st.session_state["credentials"] = []

 
    title = st.empty()
   

    #Cetak opsi nomor telp user
    with app.app_context():
        # options = list(set([f'{message.recipient}' for message in Message.query.all()]))
        options = [user.phone_number for user in User.query.all()]
        selected_number = st.selectbox('Filter dengan Nomor Telp:', options)
        user = User.query.filter_by(phone_number=selected_number).first()

    with title:
        name = user.nick_name if user is not None else ""
        st.title(f"Chat History: {name} [{selected_number}]")
 

        
         
    with st.sidebar:
        clear_chat_histories(selected_number)



    chat_history_expander = st.expander("💬 Chat History", expanded=True)
    chat_history_expander.empty()

    user_input = st.chat_input(placeholder="Ketik pesan disini ...",)

    #with st.expander("💬 Chat History", expanded=True):
    with chat_history_expander:
        # if user_input:
        show_chat_histories(selected_number, user_input)


 
          
if 'logged_in' in st.session_state and st.session_state['logged_in']:
    status_login = st.session_state['logged_in']
    run()
else:
    st.error("You are not logged in!")
    login.run()