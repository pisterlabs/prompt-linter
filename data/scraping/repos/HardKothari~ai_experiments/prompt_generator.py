import streamlit as st
from langchain import LLMChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from prompt_templates_json import PROMPTS
from langsmith.run_helpers import traceable
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter

import os
from dotenv import load_dotenv

load_dotenv()

def main_expression(openai_api_key, text, instructions, gpt_answer, prompt_key):
    model = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k", temperature=0.5)
   
    if instructions:
        special_instructions = f"SpecialInstructions: {instructions}"
    else:
        special_instructions = ""

    prompt1=PROMPTS[prompt_key]      
    
    chain1 = prompt1 | model | StrOutputParser()

    prompt_1 = chain1.invoke({"text":text})

    if gpt_answer:
        prompt2 = ChatPromptTemplate.from_template(template=f"{prompt_1}\n\n{{instructions}}")

        chain2 = {"instructions": itemgetter("instructions")} | prompt2 | model | StrOutputParser()
        # chain2 = {"prompt": prompt_1, "instructions": itemgetter("instructions")} | prompt2 | model | StrOutputParser()
        
        answer = chain2.invoke({"instructions": special_instructions})
    else:
        answer = ""    
    
    return prompt_1, answer
   

def main():

    openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key (mandatory)", type="password")
    
    st.header("Genius Prompt / Answer Generator")

    lazy_prompt = st.text_input("Enter your lazy prompt")
    
    special_instructions = st.text_input("Enter any special instructions (optional)")

    gpt_answer = st.sidebar.checkbox("GPT answer for prompt")

    prompt_key = st.sidebar.selectbox("Select a Specific Task", options=list(PROMPTS.keys()))
    
    if openai_api_key and lazy_prompt:
        if st.button("Submit"):
            prompt, answer = main_expression(openai_api_key=openai_api_key, 
                                             text=lazy_prompt, 
                                             instructions=special_instructions, 
                                             gpt_answer=gpt_answer,
                                             prompt_key = prompt_key)
            if prompt:
                # Display Prompt and Answer
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Prompt:")
                    st.write(prompt)
                with col2:
                    if gpt_answer:
                        st.subheader("Answer:")
                        st.write(answer)
                # Clear API Key
                openai_api_key = ""
    else:
        st.warning("Please populate both the OpenAI API Key and the Lazy Prompt.")


if __name__ == "__main__":
    main()
