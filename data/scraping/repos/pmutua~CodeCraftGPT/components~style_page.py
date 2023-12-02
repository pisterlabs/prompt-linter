"""
StyleSculpt - Code Style Checker

Ensure code quality and adherence to coding standards with StyleSculpt. This feature
provides feedback on coding style, offering suggestions for improvement. By enforcing
best practices, StyleSculpt enhances code quality and consistency.
"""

import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from llm.models import chat
from prompts.code_style_check_prompt import create_coding_style_prompt

def show_style_page():
    """
    Display the StyleSculpt page with a title, description, and code input form.

    Parameters:
    - openai_api_key (str): API key for accessing the OpenAI GPT-3.5 model.
    """

    st.title("StyleSculpt - Code Style Checker")
    
    st.markdown('Ensure code quality and adherence to coding standards with StyleSculpt. '
                'This feature provides feedback on coding style, offering suggestions for '
                'improvement. By enforcing best practices, StyleSculpt enhances code quality '
                'and consistency.')

    with st.form(key="style_form"):
        refined_code = st.text_area("Enter refined code")

        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            
            st.text(f"Checking code style... ✨")
            
            chat_prompt = create_coding_style_prompt(refined_code)
            chain = LLMChain(llm=chat, prompt=chat_prompt)
            result = chain.run(refined_code=refined_code)
            st.markdown(result)