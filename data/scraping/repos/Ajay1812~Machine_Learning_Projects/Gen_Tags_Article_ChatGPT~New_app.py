#importing dependecies
import pandas as pd
import streamlit as st
import openai
import re
import numpy as np
import Retail_Hub_logo
from streamlit_searchbox import st_searchbox

# def update_first():
#     st.session_state.second = st.session_state.first

def update_second():
    st.session_state.first = st.session_state.second

openai.api_key = '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'

data = pd.read_csv('startups.csv')

headerSection = st.container()

startup_name = list(data['company_short_name'])

# First select box
selected_startup = st.selectbox('Choose an option', options =  startup_name)
st.write(selected_startup)


# st.sidebar.checkbox('Check',selected_startup)

tag_prompt = f'give me list of 10 important tags or keywords related {selected_startup} startup.'

response_tags = openai.Completion.create(
engine="text-davinci-003",
prompt= tag_prompt,
max_tokens=1024,
n=1,
temperature=0.5)

# Selecting multiple tags for articles
multi_tags = response_tags['choices'][0]['text']

text = multi_tags
pattern = re.compile(r"\d+\.(.*)")
matches = pattern.findall(text)

list_of_startup_tags = []
for match in matches:
    list_of_startup_tags.append(match.strip())

st.write(multi_tags)

a = list_of_startup_tags

selected_tag = st_searchbox("Choose Tag: ",a)

st.write(selected_tag)

