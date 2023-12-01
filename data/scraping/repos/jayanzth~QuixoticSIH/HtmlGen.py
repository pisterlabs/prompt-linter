from langchain.memory import ConversationBufferMemory
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import os
# import numpy as np




    
def get_vectorstore():
    
    embeddings = OpenAIEmbeddings()
    # test = np.load('savefile.npy',allow_pickle=True)
    # if (test):
    #     return test
    # embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    # embeddings = OpenAIEmbeddings(deployment=os.getenv("OPENAI_DEPLOYMENT_NAME"),chunk_size=1)
    # vectorstore = FAISS.from_texts(texts=text_chunks, embedding = embeddings)
    vectorstore=FAISS.load_local('Htmlgen\Vectorstore',embeddings)
    print('Vector store for Html files received')
    # vectorstore.save_local('vectorstore')
    # vs = np.array(vectorstore)
    # np.save('savefile.npy',vs)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = 'chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorstore.as_retriever(),
        memory=memory
    )
    print('Conversation chain created for Html generation')
    return conversation_chain
    
def handle_userinput(user_question):
    response = st.session_state.conversation1({'question':user_question})
    print(response)
    st.session_state.chat_history = response['chat_history']
    
    for i,message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace('{{MSG}}',message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace('{{MSG}}',message.content),unsafe_allow_html=True)
            
def get():
    
    load_dotenv()
    vectorstore = get_vectorstore()
    print('Done')
    conversation1 = get_conversation_chain(vectorstore)
    return conversation1

    
    
    
    
# if __name__=='__main__':
#     main()