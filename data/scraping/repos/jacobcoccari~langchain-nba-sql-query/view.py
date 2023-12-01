import streamlit as st
from generate_response import generate_assistant_response
from langchain.utilities import SQLDatabase


def format_memory_streamlit(role, prompt):
    st.session_state.messages.append({
        "role": role,
        "content": prompt,
    })


def save_chat_history(prompt, db):
    format_memory_streamlit("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)
    assistant_response = generate_assistant_response(prompt, db)
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
    format_memory_streamlit("assistant", assistant_response)


def main():
    st.title("NBA Playerbot") 
    db = SQLDatabase.from_uri("sqlite:///nba_roster.db", sample_rows_in_table_info=0)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Ask me any questions about NBA players")

    if prompt:
        save_chat_history(prompt, db)


if __name__ == "__main__":
    main()