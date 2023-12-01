import streamlit as st 
import os
import openai

st.header("Leet Code Solution generator in python") 
leetcode_question=st.text_area("Type Probelm in following format 1. Two Sum")
button= st.button("Fetch code")

def gen_auto_response(leetcode_question):

	openai.api_key = "API KEY"

	response = openai.Completion.create(
	  model="code-davinci-002",
	  prompt="Provide Python code for following questions \n\nQuestion: {leetcode_question} \n\nCode:\n",
	  temperature=0,
	  max_tokens=64,
	  top_p=1,
	  frequency_penalty=0,
	  presence_penalty=0,
	  stop=["\"\"\""]
	)
	print(response)
	return response.choices[0].text
	
if leetcode_question and button:
    with st.spinner("Generating Autoresponse to your review Please Wait"):
        reply=gen_auto_response(leetcode_question)
        st.write(reply)