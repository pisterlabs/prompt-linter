from langchain.llms import ctransformers
import streamlit as st

llm = ctransformers.CTransformers(
    model="llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type="llama"
)

st.title('AI Poet')
topic = st.text_input('Please write down the theme of the poem.')

if topic:
    with st.spinner('Wait for it...'):
        poet = llm.predict("Write me a poem on the subject of "+ topic)
    st.write(poet)