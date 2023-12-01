import streamlit as st
import openai
import os
import PyPDF2
from dotenv import load_dotenv, find_dotenv
import textwrap

_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ' '.join([page.extract_text() for page in reader.pages])
    return text

pdf_path = '/home/anilk/PycharmProjects/streamlit2/demo.pdf'  # Replace with your actual PDF path
pdf_content = extract_text_from_pdf(pdf_path)


def get_answer_openai(question, context, chunk_size=1024):
    try:
        # Split the context into chunks
        context_chunks = textwrap.wrap(context, chunk_size)

        for chunk in context_chunks:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Question: {question}\n\nContext: {chunk}"}
            ]

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=messages,
                max_tokens=150,
                temperature=0,
                n=1,
                stop=["\n"]
            )

            # Check if the response is relevant
            answer = response.choices[0].message['content'].strip()
            if answer and not answer.lower().startswith("i'm not sure"):  # Assuming "I'm not sure" is a placeholder for non-relevant responses
                return answer

        # If no relevant response found, return a default message
        return "I'm sorry, I wasn't able to find a relevant answer to your question in the text."

    except Exception as e:
        return f"Error: {str(e)}"

st.title("OpenAI-based Question Answering from PDF")
user_input = st.text_input("🚀 Hi human! I am your smart AI. How can I help you today? ")

if user_input:
    with st.spinner('Generating answer...'):
        answer = get_answer_openai(user_input, pdf_content)
        st.write(answer)

st.sidebar.header("Settings")
model_temperature = st.sidebar.slider("Model Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Max Tokens", 50, 300, 150)

def display_footer():
    st.markdown("---")
    st.markdown(
        """
        Made with 💙 by [Anil Korkut](https://neuralbridge.com) |
        [Privacy Policy](https://yourwebsite.com/privacy) |
        [Terms of Service](https://yourwebsite.com/terms) |
        [Contact Us](mailto:contact@yourwebsite.com)
        """,
        unsafe_allow_html=True
    )
display_footer()
