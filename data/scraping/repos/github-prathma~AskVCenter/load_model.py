from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from load_data import *
import streamlit as st

def getModel(apiKey, modelName):
    return ChatOpenAI(temperature=0.7, openai_api_key=apiKey, model_name=modelName)
    

def getQAChain(llm):
    if "buffer_memory" not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

    return load_qa_chain(
        llm,
        chain_type="stuff",
        memory=st.session_state.buffer_memory,
        prompt=PromptTemplate(
            input_variables=["chat_history", "human_input", "context", "tone", "persona", "category"],
            template=
            """
            You are a chatbot who acts like {persona}, having a conversation with a human.
            Given the following extracted parts of a long document and a question, 
            Create a final answer with references ("SOURCES") in the tone {tone} on the basis of given {category}. 
            If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
            If the question from another category is asked, just say that questions asked to relevant category {category} can only be answered and 
            please select another category to get answers specific to selected category.
            ALWAYS return a "SOURCES" part in your answer.\n
            SOURCES should only be hyperlink URLs which are genuine and not made up.
            
            {context}
            {chat_history}
            
            Human: {human_input}
            Chatbot:
            
            """,
            ), verbose=False
        )