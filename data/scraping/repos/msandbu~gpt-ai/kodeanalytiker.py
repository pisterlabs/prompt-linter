import os
os.environ["OPENAI_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""
import streamlit as st
import openai
from collections import deque
from typing import Dict, List, Optional, Any
#do this to load the env variables


# Definer Streamlit layout
st.title("Kodetåke")
language = st.selectbox("Velg Språk", ["Python", "JavaScript", "PowerShell"])
code_input = st.text_area("Last opp kode som skal analyseres")


def explain_code(input_code, language):
    model_engine = "gpt-3.5-turbo" # Change to the desired OpenAI model
    message = [
                {
            "role": "system",
            "content": "You are a helpful assistant to help desribe code to the user, always reply in Norwegian back to the user. Always use Norwegian"
        },
        {
            "role": "user",
            "content": f"Forklar hva følgende {language} kode gjør for noe: \n\n{input_code}"
        }
    ]
    response = openai.ChatCompletion.create(
            model=model_engine, 
            messages=message, 
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )
    return response.choices[0].message['content']


# Temperature and token slider
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)
tokens = st.sidebar.slider(
    "Tokens",
    min_value=64,
    max_value=2048,
    value=256,
    step=64
)
# Define Streamlit app behavior
if st.button("Forklar"):
    output_text = explain_code(code_input, language)
    st.write("Kodetåke:", output_text)
