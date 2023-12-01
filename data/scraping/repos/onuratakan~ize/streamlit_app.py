import streamlit as st
import pandas as pd
import openai
import numpy as np
import altair as alt
import matplotlib.pyplot as plt


def file_reader(file):
    file_type = file.name.split('.')[-1]

    # TODO add more file types
    if file_type == 'csv':
        return pd.read_csv(file)

    if file_type == 'parquet':
        return pd.read_parquet(file_type)

    return None

# Get gpt-3 key
f = open("key.txt", "r")
key = f.read()
f.close()
openai.api_key = key

f = open("analyze_prompt.txt", "r")
analyze_prompt = f.read()
f.close()

f = open("visualize_prompt.txt", "r")
visualize_prompt = f.read()
f.close()


st.title('IZE')
st.subheader('Use natural language to visualize and analyze your data')

uploaded_file = st.file_uploader("Choose a file to analyze")


if uploaded_file is not None:
    df = file_reader(uploaded_file)

    if df is None:
        st.write(f'{uploaded_file.name.split(".")[-1]} is not a supported file type')


if uploaded_file is not None and df is not None:

    st.dataframe(df.head())

    example = 'Ex: How many rows are there?'
    user_input = st.text_input("Input your Query here", example)

    if user_input and user_input != example:
        with st.spinner('Loading...'):

            if 'graph' in user_input.lower() or 'chart' in user_input.lower() or 'plot' in user_input.lower():
                new_prompt = f"{visualize_prompt}\n\nQ:{user_input} columns: {list(df.columns)}\n"

                response = openai.Completion.create(engine='davinci',
                                                    prompt=new_prompt,
                                                    stop='\n',
                                                    temperature=0,
                                                    top_p=1,
                                                    frequency_penalty=0,
                                                    presence_penalty=0,
                                                    max_tokens=150
                                                    )
                command = f"output = {response.choices[0].text.replace('A: ', '')}; dtype = type(output)"
                print("command:", command)
                ldict = {}
                exec(command, globals(), ldict)




            else:
                new_prompt = f"{analyze_prompt}\n\nQ:{user_input} columns: {list(df.columns)}\n"

                response = openai.Completion.create(engine='davinci',
                                                    prompt=new_prompt,
                                                    stop='\n',
                                                    temperature=0,
                                                    top_p=1,
                                                    frequency_penalty=0,
                                                    presence_penalty=0,
                                                    max_tokens=150
                                                    )

                output = ''
                command = f"output = {response.choices[0].text.replace('A: ', '')}; dtype = type(output)"
                print("command:", command)
                ldict = {}
                exec(command, globals(), ldict)

                print('output:', ldict['output'])
                print('type:', ldict['dtype'])
                output = ldict['output']
                dtype = ldict['dtype']

                if dtype == pd.core.series.Series and len(output) == 1:
                    command = f"output = {response.choices[0].text.replace('A: ', '')}.iloc[0]"
                    exec(command)
                    output = int(output)

                if dtype in [int, float, np.float, np.float64, np.float32, np.int, np.int64, np.int32]:
                    st.text(output)

                if dtype in [pd.core.frame.DataFrame, pd.core.series.Series]:
                    st.dataframe(output)


