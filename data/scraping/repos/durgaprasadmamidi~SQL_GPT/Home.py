#!/usr/bin/env python3

import streamlit as st 
import openai
from utils import *
from dotenv import load_dotenv
import os

import pygments
from pygments import highlight, lexers, formatters
from pygments.lexers import SqlLexer
from pygments.formatters.html import HtmlFormatter


def main():
    """
    Main Function
    """

    st.set_page_config(
        page_title="SQLGPT",
        # page_icon="./assets/favicon.png",
        layout= "centered",
        initial_sidebar_state="expanded",
        menu_items={
        'About': "## A minimalistic application to generate SQL queries using Generative AI built with Python and Streamlit"
        } )
    st.title("SQL GPT")
    hide_footer()
    hide_hamburger_menu()
    if "cache" not in st.session_state:
        st.session_state["cache"] = []
    

    query = st.text_area("Enter the desired question to generate SQL Query")
    # openai.api_key = st.secrets["openai_api_key"]
    load_dotenv()
    openai.api_key = os.environ["openai_api_key"]
    prompt = f"Translate this natural language query into syntactically correct SQL:\n\n{query}\n\nSQL Query:"
    chb = st.checkbox("Table Schema")
    if chb:
        schema = st.text_area("Enter Table Schema")
        prompt = f"Translate this natural language query into syntactically correct SQL:\n\n{query}\n\nUse this table schema:\n\n{schema}\n\n{prompt}"
    model_engine = "text-davinci-003"
    

    if st.button("✍️ Generate SQL Query"):
        try:

            print(prompt)
            if prompt == "Translate this natural language query into syntactically correct SQL:\n\n\n\nSQL Query:":
                st.warning("Please enter a query to generate SQL")
                return
            if prompt == "Translate this natural language query into syntactically correct SQL:\n\n\n\nUse this table schema:\n\n\n\nTranslate this natural language query into syntactically correct SQL:\n\n\n\nSQL Query:":
                st.warning("Please enter a table schema to generate SQL")
                return

            completion = openai.Completion.create(
                        engine=model_engine,
                        prompt=prompt,
                        max_tokens=2048,
                        n=1,
                        stop = "\\n",
                        temperature=0.5,
                        frequency_penalty = 0.5,
                        presence_penalty = 0.5,
                        logprobs = 10)

            response = completion.choices[0].text
            st.balloons()
            st.markdown("### Output:")
            # st.text(f"{response}")
            print(response)
            formatter = HtmlFormatter(style='colorful')
            formatted_sql = highlight(response, lexers.SqlLexer(), formatter)
            st.markdown(formatted_sql, unsafe_allow_html=True)

            if response:  # If response is not empty
                st.session_state["cache"].append(response)

        except Exception as e:
            st.error(f"Error: {e}")

    if st.button('🕛 Show History'):
        if st.session_state["cache"]:
            for i, res in enumerate(st.session_state["cache"], start=1):
                st.text(f'Query Result {i}:')
                st.text(res)
        else:
            st.info("No query history available.")
    print(st.session_state["cache"])

    if st.button("🧹 Clear Cache"):
        st.session_state["cache"] = []
        st.success("Cache cleared successfully!")

    


if __name__ == "__main__":
    main()