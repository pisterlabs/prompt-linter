import streamlit as st
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set Hugging Face API token
import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

# Function to generate the pet name
def generate_pet_name(input_str):
    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = LLMChain(prompt=prompt,
                     llm=HuggingFaceHub(repo_id="MBZUAI/LaMini-Flan-T5-248M",
                                        model_kwargs={"temperature": 0.6,
                                                      "max_length": 64}))

    response = llm_chain.run(input_str)

    return response

# Streamlit app
st.title("Model")

# Input fields
input_str = st.text_area("Enter your text")

if st.button("Generate Pet Name"):
    if input_str:
        output = generate_pet_name(input_str)
        st.success(f"Model Output: {output}")
    else:
        st.warning("Please enter valid input")
