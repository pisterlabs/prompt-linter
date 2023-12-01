import streamlit as st
import os
import chromadb
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from ingest import create_vector_db
from chain import get_conversation_chain
from chat_ui import message_display, reset_chat_history

def load_chain():
  collection_name = 'pdf_data'
  dir_name = 'db'

  if not os.path.exists(dir_name):
    raise Exception(f"{dir_name} does not exist, nothing can be queried")

  client_settings = chromadb.config.Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=dir_name,
    anonymized_telemetry=False
  )

  embeddings = OpenAIEmbeddings()

  db = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    client_settings=client_settings,
    persist_directory=dir_name,
  )

  return get_conversation_chain(db)

chain = load_chain()

def execute_chain(query):
  chain_result = None
  try:
      chain_result = chain(query)
  except Exception as error:
      print("error", error)

  return chain_result

def main():
  st.set_page_config(
    page_title="PDF Bot",
    page_icon=":robot_face:",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
      'Report a bug': "https://github.com/ibizabroker/gpt-pdf-bot",
      'About': '''PDF Bot is a chatbot designed to help you answer questions from pdfs. It is built using OpenAI's GPT, chromadb and Streamlit. 
               To learn more about the project go to the GitHub repo. https://github.com/ibizabroker/gpt-pdf-bot 
               '''
    }
  )

  st.title("PDF Bot")
  st.caption("Easily chat with pdfs.")

  messages_container = st.container()

  if "generated" not in st.session_state:
    st.session_state["generated"] = ["Hey there, I'm PDF Bot, ready to chat up on any questions you might have regarding the documents you have uploaded."]
  if "past" not in st.session_state:
    st.session_state["past"] = ["Hey!"]
  if "input" not in st.session_state:
    st.session_state["input"] = ""
  if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

  if "messages" not in st.session_state:
    st.session_state["messages"] = [("Hello! I'm a chatbot designed to help you with pdf documents.")]

  c1, c2, c3 = st.columns([6.2, 1, 1])
  with c1:
    query = st.text_input(
      label="Query", 
      key="input", 
      value="", 
      placeholder="Ask your question here...",
      label_visibility = "collapsed"
    )
  
  with c2:
    submit_button = st.button("Submit")

  with c3:
    reset_button = st.button("Reset")

  if reset_button:
    reset_chat_history()

  if len(query) > 1 and submit_button:
    messages = st.session_state['messages']

    result = execute_chain(query)

    for i, message in enumerate(result['chat_history']):
      if i % 2 == 0:
        st.session_state.past.append(message.content)
        # print("user:" + message.content)
      else:
        messages.append((query, message.content))
        st.session_state.generated.append(message.content)
        # print("bot:" + message.content)

  with messages_container:
    if st.session_state["generated"]:
      for i in range(len(st.session_state["generated"])):
        message_display(st.session_state["past"][i], is_user=True)
        message_display(st.session_state["generated"][i])

  with st.sidebar:
    st.subheader("Your documents")
    pdfs = st.file_uploader(
      "Upload your PDFs here and click 'Upload to DB'", 
      type=['pdf'],
      accept_multiple_files=True
    )
    if pdfs is not None:
      for pdf in pdfs:
        with open(pdf.name, "wb") as f:
          f.write(pdf.getbuffer())

    if st.button("Upload to DB"):
      with st.spinner("Processing"):
        vector_store = create_vector_db()

        st.session_state.conversation = get_conversation_chain(vector_store)
        st.write("Processing Done")

  hide_footer = """
                  <style>
                    footer {visibility: hidden;}
                  </style>
                """
  st.markdown(hide_footer, unsafe_allow_html=True) 

if __name__ == '__main__':
  main()