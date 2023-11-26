import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
from langchain import OpenAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain import agents
from langchain.docstore.document import Document
from nltk.tokenize import sent_tokenize
import nltk
import os

load_dotenv()
nltk.download('punkt')  # Download the NLTK Punkt tokenizer if it's not already present
from pdf_jarvis import main as pdf_jarvis_main

# Function to scrape the given website
def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    else:
        raise Exception(f"Unable to access website with status code: {response.status_code}")

def split_text(text, max_tokens):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len(" ".join(current_chunk)) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def get_summary(text):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    
    max_tokens = 4097 - 256
    text_chunks = split_text(text, max_tokens)

    summaries = []
    for chunk in text_chunks:
        doc = Document(page_content=chunk)
        summary = chain.run([doc])
        summaries.append(summary)

    return summaries

def main():
    st.title("Website Summarizer and PDF Inquirer")

    page = st.sidebar.selectbox(
        "Choose a page",
        ("Website Scraper and Summarizer", "PDFJarvis"),
    )

    if page == "Website Scraper and Summarizer":
        st.subheader("This app uses Langchain and OpenAI to scrape a website and summarize it using GPT-3.")
        st.write("Please enter a website URL and the app will scrape, then chunk the data and send to OpenAI to summarize.")

        with st.sidebar:
            url_input = st.text_input("Enter the website URL:")
            summarize_button = st.button("Summarize with GPT", key="summarize")

        extracted_text = None

        if summarize_button:
            if not url_input:
                st.error("Please enter a website URL.")
            else:
                with st.spinner("Processing"):
                    if not extracted_text:
                        try:
                            extracted_text = scrape_website(url_input)
                        except Exception as e:
                            st.error(f"Error: {e}")

                    try:
                        summaries = get_summary(extracted_text)
                        combined_summary = " ".join(summaries)
                        st.write(combined_summary)
                    except Exception as e:
                        st.error(f"Error: {e}")

    elif page == "PDFJarvis":
        pdf_jarvis_main()

if __name__ == "__main__":
    main()
