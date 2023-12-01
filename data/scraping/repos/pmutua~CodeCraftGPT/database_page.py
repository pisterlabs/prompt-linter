"""
This part of the code show the AI part of database code.
"""

import streamlit as st
from langchain.chains import LLMChain
from data.database_System import Database_system
from llm.models import chat
from prompts.translate_code_prompt import create_translation_prompt

def show_database_page():
    """
    Displays and generate the Database query.

    Parameters:
    - openai_api_key (str): The API key for OpenAI.

    Returns:
    None
    """
    st.title("Database - Database Query Question")

    st.markdown('Database AI Tool ')

    with st.form(key="lang_form"):
        source_code = st.text_area("Enter Your Question to Generate the Query")
        target_language = st.selectbox("Select Select Dasabase You Want", Database_system)

        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            
            st.text(f"Translating code snippet to {target_language}................✨")

            chat_prompt = create_translation_prompt(target_language,source_code)

            chain = LLMChain(llm=chat, prompt=chat_prompt)
            result = chain.run(source_code=source_code,
                               target_language=target_language)
            
            st.text_area("Translated Code", result, height=400)
            