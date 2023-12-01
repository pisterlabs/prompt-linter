import streamlit as st
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables and set up the OpenAI API key
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Set up the LLM, API Key will be rotated so input your own key
llm = OpenAI(openai_api_key=openai_api_key , temperature=1.2, max_tokens=900)

def main():
    st.title("Text Musical Generator in Different Languages")

    # Dropdown menu with six languages
    languages = ["English", "Spanish", "French", "German"]
    selected_language = st.selectbox("Choose a response language:", languages)

    # Modify the prompt to instruct the model to explain the user's input in a musical style in the desired language
    language_instructions = {
        "English": "",
        "Spanish": "español",
        "French": "français",
        "German": "Deutsch"
    }
    instruction = language_instructions[selected_language]
    prompt_template = f"Create a script for a {instruction} musical using this statement: '{{input}}'"

    prompt = PromptTemplate(input_variables=["input"], template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)

    user_input = st.text_input("Enter your prompt (in English):")
    
    if user_input:
        response = chain.run({
            'input': user_input
        })
        st.text_area("Response:", value=response, height=250, max_chars=1200, key=None)

if __name__ == "__main__":
    main()
