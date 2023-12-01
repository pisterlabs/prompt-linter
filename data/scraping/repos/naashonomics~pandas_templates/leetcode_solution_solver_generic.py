import streamlit as st 
import os
import openai

st.header("Leet Code Solution generator in python") 
language_option = st.selectbox(
    'In which language do you want your leetcode solution?',
    ('Python', 'Java', 'C++','Javascript','Go','Ruby'))
leetcode_question=st.text_area("Type Probelm in following format <type Leet Code with constraints> ")
button= st.button("Fetch code")

def gen_auto_response(leetcode_question,language_option):

    openai.api_key = "API KEY"
    response = openai.Completion.create(
        model="code-cushman-001",
        prompt=f""""Given a {language_option} solution for the leetcode question below 
                Leet Code Question: {leetcode_question} 
                {language_option} Solution: """,
        temperature=0,
        max_tokens=1124,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    print(response)
    return response.choices[0].text
	
if leetcode_question and button and language_option:
    with st.spinner("Generating Autoresponse to your review Please Wait"):
        reply=gen_auto_response(leetcode_question,language_option)
        st.code(reply)