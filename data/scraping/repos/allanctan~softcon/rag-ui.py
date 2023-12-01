import tiktoken
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

from dotenv import load_dotenv

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            
def read_file(filename):
  with open(filename, 'r') as file:
    text = file.read()
  return text

def main():
  load_dotenv()
  if "conversation" not in st.session_state:
      st.session_state.conversation = None
      encoding = tiktoken.get_encoding("cl100k_base")
      # text below is 75 words 
      long_text = read_file('constitution.txt')
      encoded = encoding.encode(long_text)
      print('Tiktoken length:',len(encoded)) # returns 95

      text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, 
                                                     chunk_overlap=20, 
                                                    separators=['\n\n','\n', ' '  ,''])
      chunks = text_splitter.split_text(long_text)
      print('Chunks length:',len(chunks))
      embeddings = OpenAIEmbeddings()
      vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
      st.session_state.conversation = get_conversation_chain(vectorstore)

  if "chat_history" not in st.session_state:
      st.session_state.chat_history = None

  st.set_page_config(page_title="Chat about Philippine Constitution",
                      page_icon=":books:")
  st.write(css, unsafe_allow_html=True)
  st.header("Chat about Philippine Constitution :books:")

  user_question = st.text_input("What is your question?")
  if user_question:
      handle_userinput(user_question)
 
if __name__ == '__main__':
  main()