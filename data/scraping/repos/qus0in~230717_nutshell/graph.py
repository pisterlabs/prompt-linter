import streamlit as st
import openai

def system(msg):
    return {"role" : "system", "content": msg}
def user(msg):
    return {"role" : "user", "content": msg}

def get_mermaid(extracted_text):
    openai.api_key = st.secrets["openai_api_key"] # API Key
    messages = [
        system("I'm going to give you text extracted from a PDF textbook used to teach Python."),
        system("Based on what you've been handed, give us a bullet point nutshell."),
        system("Write a tree-structured mermaid markdown of the cleaned up nutshell in order of importance and relationship."),
        system("I want it to cover important concepts in Python programming and have more than two levels of depth."),
        user(extracted_text),
        system("Reduce to 1200 characters or less without sacrificing meaning."),
        system("The output should only receive markdown wrapped in ```, and the contents of mermaid should be written in Korean."),
        system("Don't show me any results other than mermaid."),
    ]
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return completion['choices'][0]['message']['content']

def handle_graph():
    if "extracted_text" in st.session_state:
        if st.button("NutShell 생성하기"):
            with st.spinner("생성하는 중"):
                mermaid = get_mermaid(st.session_state.extracted_text)
            with st.expander("🧜‍♀️ Mermaid Markdown"):
                st.write(mermaid)