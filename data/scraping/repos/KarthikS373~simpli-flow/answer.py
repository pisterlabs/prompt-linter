import streamlit as st
import openai
import os

from dotenv import load_dotenv

load_dotenv()


# Set your OpenAI API key here
openai.api_key = os.getenv('API_KEY')


def answer_question(description):
    description_fin = "\nYou are an expert in developing Smart Contracts in the Cadence Language. Answer the following question concisely:"+description 
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=description_fin,
        max_tokens=1000,
        temperature=0.7,
        n=1,
        stop=None
    )

    code = response.choices[0].text.strip()
    return code

def main():
    st.title("Blockchain Question Answerbot")
    code_input = st.text_area("Enter question:", height=200)
    generate_button = st.button("Generate Answer")
    code_output = st.empty()

    if generate_button:
        code = answer_question(code_input)
        code_output.text(code)

if __name__ == "__main__":
    main()
