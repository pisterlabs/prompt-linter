import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from utils.authenticate import authenticate
from configs.apikey import apikey
import os

os.environ["OPENAI_API_KEY"] = apikey  

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string


def page_chat():
    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["How can I assist you?"]
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    if 'buffer_memory' not in st.session_state:
                st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


    embedding = OpenAIEmbeddings()

    vector_store = Chroma(
        collection_name="fusion-ai",
        embedding_function=embedding,
        persist_directory="db/chroma",
    )
    response_container = st.container()
    textcontainer = st.container()
    with st.expander("Options"):
        chain_type = st.selectbox(label="Chain Type",options=["stuff","refine","map_reduce","map_rerank"],index=0) 

    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(llm, chain_type=chain_type)
    conversation = ConversationalRetrievalChain(retriever=vector_store.as_retriever(), verbose=False, return_source_documents=True,question_generator=question_generator,combine_docs_chain=doc_chain)
    chat_history = []

    with textcontainer:
        query = st.text_input("Query: ", key="input")
        if query:
            with st.spinner("Fetching..."):
                response = conversation({"question": query, "chat_history": chat_history})
                answer = response["answer"]
                with st.expander("Source Documents"):
                    st.write(response["source_documents"])
                chat_history.append(HumanMessage(content=query))
                chat_history.append(AIMessage(content=answer))
            st.session_state.requests.append(query)
            st.session_state.responses.append(answer) 
    with response_container:
        if st.session_state['responses']:

            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i],key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')


auth  =  authenticate()
if auth[0]:
    page_chat()