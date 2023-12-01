import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks import get_openai_callback

from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

class EmailChatbot:

    def __init__(self, model_name, temperature, vectors):
        """
        Método construtor que inicializa algumas variáveis do chat.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.vectors = vectors

    system_template = """
        You are a helpful AI email assistant. You will receive a file with the 
        user emails of the past 60 days which is represented by the following 
        pieces of context, use those emails and the chat history to answer the user question.
        If the user asks you to reply a email: Reply in the same tone and context 
        of the last emails between the user and the person for who the email will 
        be sent (if exists). Besides that, reply as if you are the user, using his 
        name, tone and way of speaking and other informations provided by the user in the question.
        If the user asks a specific question: Go through his emails to find the 
        information. If the user specifies the sender use just messages sent by that sender.
        Keep in mind that when a subject starts with 'Re:' this email is a reply for a previous one.
        If the user asks about his tasks: Go through his emails and extract all things that he said that will do or that that the senders asked him to. 
        If you don't know the answer, just say you don't know. Do NOT try to 
        make up an answer.
        If the question is not related to the context, politely respond that you 
        are tuned to only answer questions that are related to the context.
        Use as much detail as possible when responding.

        context: {context}
        """
    
    user_template = "Question: '''{question}'''"

    messages = [SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template(user_template)]
    
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    def conversationalChat(self, query):
        """
        Cria um chat usando a biblioteca LangChain
        """
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        retriever = self.vectors.as_retriever()

        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, chain_type="stuff", combine_docs_chain_kwargs={'prompt': self.qa_prompt})

        chainInput = {"question": query, "chat_history": st.session_state["history"]}
        result = chain(chainInput)

        st.session_state["history"].append((query, result["answer"]))
        return result["answer"]