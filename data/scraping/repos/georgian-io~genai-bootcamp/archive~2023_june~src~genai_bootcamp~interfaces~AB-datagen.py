import csv
import hashlib
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI


def main(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    st.title('Langchain LLM App')

    # Model name
    model_name = st.text_input('Model Name:', value='gpt-3.5-turbo')

    # Inputs for prompt templates
    prompt_template1 = st.text_input('Prompt Template 1:', value='re-write this as an Eminem rap: {input}')
    prompt_template2 = st.text_input('Prompt Template 2:', value='re-write this as an Jay-Z rap: {input}')

    # Input for the string to be inserted in the prompt
    input_string = st.text_input('Input String:', value='I went to the store, bought doritos, a drink and chewing gum')

    # Submit button
    submit_button = st.button('Submit')

    # LLM
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=0.5,
    )

    if submit_button and prompt_template1 and prompt_template2 and input_string:
        # Calculate the hashes of the prompt templates
        hex_dig1 = model_name + '_' + hashlib.md5(prompt_template1.encode()).hexdigest()
        hex_dig2 = model_name + '_' + hashlib.md5(prompt_template2.encode()).hexdigest()

        # CSV file names
        csv_path_1 = output_dir / f"{hex_dig1}.csv"
        csv_path_2 = output_dir / f"{hex_dig2}.csv"

        # Use langchain to combine prompt template and input and run through LLM
        output_string1 = LLMChain(llm=llm, prompt=PromptTemplate(
            input_variables=['input'], template=prompt_template1)).run(input=input_string)
        output_string2 = LLMChain(llm=llm, prompt=PromptTemplate(
            input_variables=['input'], template=prompt_template2)).run(input=input_string)

        col1, col2 = st.columns(2)
        with col1:
            st.header('Output 1')
            st.write(output_string1)
        with col2:
            st.header('Output 2')
            st.write(output_string2)

        # Serialize the interactions
        serialize_interaction(csv_path_1, input_string, output_string1)
        serialize_interaction(csv_path_2, input_string, output_string2)

        st.success('Interactions serialized!')


def serialize_interaction(csv_path: Path, input_string, output_string):
    df = pd.DataFrame()
    # Check if CSV file exists
    if csv_path.exists():
        # If it exists, open in append mode
        df = pd.read_csv(csv_path)

    row_df = pd.DataFrame([[input_string, output_string]], columns=['Input', 'Output'])
    df = pd.concat([df, row_df])
    df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    main(output_dir='./notebooks/day-2/outputs')
