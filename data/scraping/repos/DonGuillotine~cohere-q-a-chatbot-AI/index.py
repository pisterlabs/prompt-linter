import streamlit as st
from main import CoHere

st.header('A Cohere Powered Application')

api_key = st.text_input('OpenAI API Key:', type='password')

st.header('Your Personal chat bot - Donald!')

question_for_donald = st.text_input('Question for Donald')

cohere = CoHere(api_key)

if st.button('Answer'):
    st.write(cohere.cohere(question_for_donald))