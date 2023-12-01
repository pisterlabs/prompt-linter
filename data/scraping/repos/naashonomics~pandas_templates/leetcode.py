import streamlit as st 
import os
import openai

st.header("Leet Code Solution generator in python") 
leetcode_question=st.text_area("Type Problem in following format <type Leet Code with constraints> ")
button= st.button("Fetch Python code")

def gen_auto_response(leetcode_question):

    openai.api_key = "Enter your OPEN AI API key"
    response = openai.Completion.create(
        model="code-cushman-001",
        prompt=f""""Given a Python solution for the leetcode question below 
                Leet Code Question: {leetcode_question} 
                Python Solution: """,
        temperature=0,
        max_tokens=1124,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    print(response)
    return response.choices[0].text
	
if leetcode_question and button:
    with st.spinner("Generating Python Solution to your leetcode problem Please Wait"):
        reply=gen_auto_response(leetcode_question)
        st.code(reply)
