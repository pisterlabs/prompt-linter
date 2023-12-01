import streamlit as st

st.header("Caption Generator")
st.write("ไอเดียการเขียนแคปชั่นประกอบโพส")
st.sidebar.header("Caption Generator")
openai_api_key = st.sidebar.text_input("Please add your OpenAI API key to continue", "")

if openai_api_key:
    st.sidebar.success('OpenAI API key provided!', icon='✅')
else:
    st.sidebar.warning('Please enter your OpenAI API key!', icon='⚠️')


import openai

chatbot_input = st.text_input("ต้องการเขียนแคปชั่นเกี่ยวกับอะไร")

if openai_api_key and chatbot_input:
    openai.api_key = openai_api_key
    response = openai.Completion.create(
      engine="text-davinci-002",
      prompt=chatbot_input,
      temperature=0.5,
      max_tokens=100
    )
    st.text_area("Response:", response.choices[0].text.strip())
