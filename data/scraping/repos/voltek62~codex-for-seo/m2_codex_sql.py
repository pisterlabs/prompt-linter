"""Streamlit app to analyze data with English language commands."""

# Import from standard library
import os
import textwrap

# Import from 3rd party libraries
import streamlit as st
import openai

# Assign OpenAI API key from environment variable or streamlit secrets dict
openai.api_key = st.secrets["API_TOKEN"]

from streamlit.scriptrunner.script_run_context import get_script_run_ctx
from streamlit.server.server import Server

def _get_session():
    session_id = get_script_run_ctx().session_id
    session = Server.get_current().get_session_by_id(session_id)
    if session is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return str(session)[50:64]

def openai_call(prompt: str, stop: str = None) -> str:
    """Call OpenAI Codex with text prompt.
    Args:
        prompt: text prompt
        stop: stop sequence to interrupt further token generation
    Return: predicted response text
    """
    # Pass a uniqueID for every user w/ each API call (both for Completion & the Content Filter) e.g. user= $uniqueID. 
    # This 'user' param can be passed in the request body along with other params such as prompt, max_tokens etc.
    #uniqueID = _get_session()
    
    kwargs = {
        "engine": "code-davinci-002",
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "best_of": 1
        #"user": uniqueID,
    }
    if stop:
        kwargs["stop"] = stop
    response = openai.Completion.create(**kwargs)
    print(response)
    return response["choices"][0]["text"]

# Page start

# region main

def main():

    text_input_container = st.empty()
    password03 = text_input_container.text_input("")

    if not password03:
        st.warning("Please type your password!")
        st.stop()
    elif "M2S4SQL" in password03:
        text_input_container.empty()
    else:
        st.warning("🚨 Nope, this password doesn't work")
        st.stop()

    st.title("English to SQL")

    with st.form(key="myform"):
        table_name_label = "Table name"

        table_name = st.text_input(label=table_name_label, value="traffic")
        column_names = st.text_area(
            label="Column names (comma-separated; optionally specify data types in parentheses)",
            value="url (string), event (string), country (string)",
        )
        statement = st.text_area(
            label="English text prompt/query statement",
            value="Count the number of pageview events by url for urls with at least 10 pageviews",
        )
    
        if statement:

            prompt = textwrap.dedent(
                f'''
                """
                The database table "{table_name}" contains the following colums: {column_names}
                """
    
                # {statement}
                sql = """
                '''
            )
            stop = '"""'
            result_prefix = ""
            language = "sql"
    
            submitted = st.form_submit_button(label="Execute")
    
            if submitted:            
                st.header("Result")
                
                col1, col2, col3 = st.columns(3)
                with col2:
                    gif_runner = st.image("images/mouse.gif")
                    
                st.code(result_prefix + openai_call(prompt, stop), language=language)
                
                gif_runner.empty()
        
                st.markdown("""---""")
                st.header("Prompt sent to Codex")
                st.text(prompt)


# endregion main


if __name__ == "__main__":
    main()
