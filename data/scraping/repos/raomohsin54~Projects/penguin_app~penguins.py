import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

st.title("Palmer's Penguins")

# import our data
penguins_df = pd.read_csv(r'C:\Users\mmukhtiar\Downloads\penguins.csv')
st.write(penguins_df.head())

# get user input prompt
prompt = st.text_input("Enter your prompt:")

# initialize PandasAI and OpenAI
llm = OpenAI()
pandas_ai = PandasAI(llm)

# run PandasAI with user input prompt
pandas_ai.run(penguins_df, prompt=prompt)


