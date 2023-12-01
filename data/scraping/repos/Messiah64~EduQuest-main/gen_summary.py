import openai
import streamlit as st 

@st.cache_data
def create_content(transcript):

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Create a detailed summary, with main points as chapter headings and point form details of it, in markdown format for the given transcript\n{transcript}\n",
        temperature=0.7,
        max_tokens=600,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    markdown_content = response["choices"][0]["text"]
    return markdown_content
