import streamlit as st
from decouple import config
from langchain import PromptTemplate, LLMChain
from langchain import HuggingFaceHub
import os

os.environ['HUGGINGFACEHUB_API_TOKEN'] = config('HF_API_KEY')

st.markdown("# Ask question to google/flan-t5-xxl🎉")

repo_id = "google/flan-t5-xxl"

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64})


template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])


query = st.text_input("Ask questions and get answer from web :")
if query:
    st.markdown("## Question")
    st.write(query)
    st.markdown("## Answer")
    with st.spinner('Loading model...'):
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        st.write(llm_chain.run(query))
