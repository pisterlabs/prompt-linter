import json
import streamlit as st
import pandas as pd
from langchain import OpenAI
from oai_utils import get_openai_api_key

openai_api_key = get_openai_api_key()

def generate_response(input_text, model_name):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key, model_name=model_name)
    st.info(llm(input_text))

st.markdown("# Spreadsheet LLM")
st.sidebar.markdown("# Spreadsheet LLM")


uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    edited_df = st.data_editor(df, num_rows="dynamic")

    with st.form('my_form'):
        oai_model = st.selectbox(
            'Which OpenAI model should we use?',
            ('gpt-3.5-turbo', 'gpt-4', 'ada', 'babbage', 'curie', 'davinci'),
        )
        text = st.text_area('Prompt:', '', placeholder='How do I use Streamlit to query OpenAI?')
        submit_row = st.form_submit_button('Query for first row')
        submit_all_rows = st.form_submit_button('Query for all rows')

        if submit_row:
            row = edited_df[:1].to_json()
            full_prompt = f"""
            You are answering questions about this spreadsheet row encoded as JSON: {row}

            {text}
            """
            generate_response(full_prompt, oai_model)

        if submit_all_rows:
            for i in edited_df.index:
                row = edited_df[i:i+1].to_json()

                st.info(row)
                full_prompt = f"""
                You are answering questions about this spreadsheet row encoded as JSON: {row}

                {text}
                """
                generate_response(full_prompt, oai_model)                
