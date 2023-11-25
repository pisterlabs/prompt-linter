import streamlit as st
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os
os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

def generate_pet_name(animal_type, pet_color):

    template = """Question: {question}

    Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = LLMChain(prompt=prompt,
                     llm=HuggingFaceHub(repo_id="MBZUAI/LaMini-Flan-T5-248M",
                                        model_kwargs={"temperature":0.6,
                                                      "max_length":64}))

    question = f"Generate a python code for counting factorial. write each line of code in seperate lines."
    response = llm_chain.run(question)

    return response

if __name__ == "__main__":
    print(generate_pet_name("cow",'black'))