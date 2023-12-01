from pdfminer.high_level import extract_text
# import streamlit as st
import openai
import regex as re
import pandas as pd
import numpy as np
import openai

openai.api_key = "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"

def questions(pdf1,i):
    question = i
    prompt = f"{question}\n{pdf1}\n\\n:"
    model = "text-davinci-003"
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=500,
        temperature=0,
        n = 1,
        stop=None
    )
    diffs = response.choices[0].text
    return diffs.strip()


def get_insight(uploaded_file,p):
    # uploaded_file = st.file_uploader("Upload Contract", "pdf")
    if uploaded_file is not None:
        element = extract_text(uploaded_file)
        answers = questions(element,p)
    return answers
       

if __name__ == "__main__":
    file="C:/Users/01934L744/Box/Baijnath Data/Project 2023/Nidhi/contract/upload_ppv/Contract1.pdf"
    p = "What is the supplier name"
    result = get_insight(file,p)
    print(result)