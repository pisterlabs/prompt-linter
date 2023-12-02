import os

import openai
import pandas as pd
import streamlit as st
from ai import Doc
from ai import summarize
from streamlit.errors import StreamlitAPIException
from table import load_data, load_config, load_datasets

try:
    st.set_page_config(page_title="App", layout="wide")
except StreamlitAPIException:
    pass

openai.api_key = os.environ["OPENAI_API_KEY"]
(_, prompts) = load_data()
config = load_config()


# Streamlit app title
st.title("Document Summarization Bot")

# Handle different input methods
input_method = st.selectbox("Select input method", ("File", "URL", "Text"))

if input_method == "File":
    uploaded_file = st.file_uploader("Upload a file (.txt)", type=["txt"])
    if uploaded_file is not None:
        input_doc = Doc.from_bytes(uploaded_file)
elif input_method == "URL":
    url = st.text_input("Enter URL")
    if url:
        try:
            input_doc = Doc.from_url(url)
        except Exception as e:
            st.write(f"Error: Unable to fetch data from the URL, {e}")
elif input_method == "Text":
    input_text = st.text_area("Enter text")
    if input_text:
        input_doc = Doc.from_string(input_text)
prompt_name = config["prompt"]

# "Summarize" button
if st.button("Summarize") and input_doc.content.strip():
    text = input_doc.content
    if input_doc.filename:
        metadata = input_doc.filename
    elif input_doc.url:
        metadata = input_doc.url
    else:
        metadata = "raw_string"
    # call AI model
    selected_prompt = next(filter(lambda x: x.name == prompt_name, prompts))
    summary = summarize(text, prompt=selected_prompt.prompt)

    st.write(f"Summary:\n {summary}")
