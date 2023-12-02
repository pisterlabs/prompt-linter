import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Neo4jVector

from handlers.message import Message, write_message
from .neo4jrag import Neo4jRAGChain
from .prompts import system_prompt, context_prompt, user_prompt

embeddings = OpenAIEmbeddings()
chat = ChatOpenAI(temperature=0, openai_api_key=st.secrets['OPENAI_API_KEY'], model=st.secrets.get('OPENAI_GPT_MODEL', 'gpt-4'))
store = Neo4jVector.from_existing_index(
    embedding=embeddings,
    index_name=st.secrets["NEO4J_VECTOR_INDEX_NAME"],
    url=st.secrets["NEO4J_HOST"],
    username=st.secrets["NEO4J_USERNAME"],
    password=st.secrets["NEO4J_PASSWORD"],
)

chain = Neo4jRAGChain(
    chat=chat,
    retriever=store.as_retriever(),
    system_prompt=system_prompt,
    context_prompt=context_prompt,
    user_prompt=user_prompt
)

def generate_response(prompt):
    message = Message("user", prompt)
    st.session_state.messages.append(message)
    write_message(message)

    with st.spinner('Thinking...'):

        history = [
            AIMessage(content=m.content) if m.role == "assistant" else HumanMessage(content=m.content) for m in st.session_state.messages[:-3]
        ]

        # result = chain({"question": prompt}, history=history)
        result = chain.call_with_history({"question": prompt}, history=history)

        response = Message("assistant", result["content"])

        st.session_state.messages.append(response)

        write_message(response)
