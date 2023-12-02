import streamlit as st
# import the function generate_assistant_response from the file geneerate_response.py
from generate_response import generate_assistant_response
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


def format_memory_streamlit(role, prompt):
    st.session_state.messages.append({
        "role": role,
        "content": prompt,
    })


def save_chat_history(prompt, retriever):
    format_memory_streamlit("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)
    assistant_response = generate_assistant_response(prompt, retriever, st.session_state.messages)
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    format_memory_streamlit("assistant", assistant_response)


def main():
    st.title("MD Anderson ChemoBot") 
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    embedding_function = OpenAIEmbeddings()
    db = Chroma(
        persist_directory="./db_chemo_guide",
        embedding_function=embedding_function,
    )
    retriever = db.as_retriever(search_type="mmr", k=4) 

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("How can I help you with your chemotherapy?")

    if prompt:
        save_chat_history(prompt, retriever)


if __name__ == "__main__":
    main()