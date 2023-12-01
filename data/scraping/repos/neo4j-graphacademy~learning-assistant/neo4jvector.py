import streamlit as st
import langchain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings

from handlers.message import Message, write_message

langchain.debug = True

embeddings = OpenAIEmbeddings()

store = Neo4jVector.from_existing_index(
    embedding=embeddings,
    index_name=st.secrets["NEO4J_VECTOR_INDEX_NAME"],
    url=st.secrets["NEO4J_HOST"],
    username=st.secrets["NEO4J_USERNAME"],
    password=st.secrets["NEO4J_PASSWORD"],
)
retriever = store.as_retriever()

llm = OpenAI(
    temperature=0,
    openai_api_key=st.secrets["OPENAI_API_KEY"],
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)


def generate_response(prompt):
    message = Message("user", prompt)
    st.session_state.messages.append(message)

    write_message(message)

    with st.spinner('Thinking...'):
        answer = qa(prompt)

        response = Message("assistant", answer["result"])
        st.session_state.messages.append(response)

        write_message(response)
