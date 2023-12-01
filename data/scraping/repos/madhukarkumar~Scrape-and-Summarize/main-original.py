import streamlit as st
import requests
from bs4 import BeautifulSoup
import openai
from langchain import OpenAI
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain import agents
from langchain.docstore.document import Document
from nltk.tokenize import sent_tokenize
import nltk
import os
nltk.download('punkt')  # Download the NLTK Punkt tokenizer if it's not already present

# Function to scrape the given website
def scrape_website(url):
    response = requests.get(url) # Send an HTTP request to the provided URL
    if response.status_code == 200: # Check if the request was successful
        soup = BeautifulSoup(response.text, 'html.parser') # Parse the HTML response using BeautifulSoup
        return soup.get_text()  # return the text content of the webpage
    else:
        raise Exception(f"Unable to access website with status code: {response.status_code}")

# Function to split the text into chunks of 4096 tokens or less
def split_text(text, max_tokens):
    sentences = sent_tokenize(text)# Split the text into sentences using NLTK's Punkt tokenizer
    chunks = [] # List to store the chunks
    current_chunk = [] # List to store the current chunk

    for sentence in sentences:
        current_chunk.append(sentence) # Add the sentence to the current chunk
        if len(" ".join(current_chunk)) >= max_tokens: # Check if the current chunk is greater than or equal to the max tokens
            chunks.append(" ".join(current_chunk)) # Add the current chunk to the list of chunks
            current_chunk = [] # Reset the current chunk

    if current_chunk: # If there's any remaining text in the current chunk
        chunks.append(" ".join(current_chunk)) # Add the current chunk to the list of chunks

    return chunks

# Function to get the summary of the text
def get_summary(text):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # Get the OpenAI API Key from the environment variables
    llm = OpenAI(openai_api_key=OPENAI_API_KEY) # Create an OpenAI object
    chain = load_summarize_chain(llm, chain_type="map_reduce") # Load the summarization chain
    
    max_tokens = 4097 - 256  # 256 tokens reserved for completion
    text_chunks = split_text(text, max_tokens) # Split the text into chunks of 4096 tokens or less

    summaries = [] # List to store the summaries
    for chunk in text_chunks: # Iterate over the chunks
        doc = Document(page_content=chunk) # Create a document object
        summary = chain.run([doc]) # Run the summarization chain on the document
        summaries.append(summary) # Add the summary to the list of summaries

    return summaries

# Main function
def main():
    st.title("Website Scraper and Summarizer")
    #create an H2 subheader
    st.subheader("This app uses Langchain and OpenAI to scrape a website and summarize it using GPT-3.")
    st.write("Please enter a website URL and the app will scrape, then chunk the data and send to OpenAI to summarize.")

    with st.sidebar: # Create a sidebar
        url_input = st.text_input("Enter the website URL:")
        #api_key_input = st.text_input("Enter your OpenAI API Key:")
        #scrape_button = st.button("Scrape", key="scrape")
        summarize_button = st.button("Summarize with GPT", key="summarize")

    extracted_text = None

    # Check if the scrape button was clicked
    #if scrape_button:
    #    if not url_input:
    #        st.error("Please enter a website URL.")
    #    else:
    #        try:
    #            extracted_text = scrape_website(url_input) # Scrape the website
    #            st.write(extracted_text) # Display the extracted text
    #        except Exception as e:
    #            st.error(f"Error: {e}")

    if summarize_button:
        if not url_input:
            st.error("Please enter a website URL.")
        #elif not api_key_input:
            #st.error("Please enter your OpenAI API Key.")
        else:
            if not extracted_text:
                try:
                    extracted_text = scrape_website(url_input) # Scrape the website
                except Exception as e:
                    st.error(f"Error: {e}")

            try:
                summaries = get_summary(extracted_text) # Get the summaries by passing the extracted text 
                combined_summary = " ".join(summaries) # Combine the summaries into a single string
                st.write(combined_summary) # Display the combined summary
            except Exception as e:
                st.error(f"Error: {e}")
    # Add custom CSS to the app
    st.markdown("""
        <style>
            .stButton>button {
                background-color: blue;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
