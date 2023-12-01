import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


class Chatbot:

    def __init__(self, model_name, temperature, vectors):
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors

    def conversational_chat(self, query):
        """
        Starts a conversational chat with a model via Langchain
        """
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name=self.model_name, temperature=self.temperature),
            memory=st.session_state["history"],
            retriever=self.vectors.as_retriever(),
        )
        result = chain({"question": query})

        return result["answer"]
