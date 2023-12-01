import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain.callbacks import StreamlitCallbackHandler
import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import json

load_dotenv()

def get_next_filename():
    i = 1
    while True:
        filename = f"/mnt/data/scraped_results_{i}.json"
        if not os.path.exists(filename):
            return filename
        i += 1

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")



schemas = {
    "WSJ": {
        "properties": {
            "news_article_title": {"type": "string"},
            "news_article_summary": {"type": "string"},
        },
        "required": ["news_article_title", "news_article_summary"],
    },
    "Another Schema": {
        # Define other schema here
    },
    # Add more schemas as needed
}

def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).run(content)

def scrape_with_playwright(urls, schema):
    
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(docs,tags_to_extract=["span"])
    print("Extracting content with LLM")
    
    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, 
                                                                    chunk_overlap=0)
    splits = splitter.split_documents(docs_transformed)
    
    # Process the first split 
    extracted_content = extract(
        schema=schema, content=splits[0].page_content
    )
    pprint.pprint(extracted_content)
    return extracted_content

st.title("Playwright Scraping Agent :computer:")


# Create columns to align the input, button, and dropdown
col1, col2, col3 = st.columns(3)
with col1:
    urls_input = st.text_input("Enter URLs (comma-separated):")

with col2:
    process_button = st.button("Process")

with col3:
    selected_schema_name = st.selectbox("Select Schema:", list(schemas.keys()))

# Retrieve URLs from user input and split by commas
if process_button:
    selected_schema = schemas[selected_schema_name]
    urls = [url.strip() for url in urls_input.split(",")]
    extracted_content = scrape_with_playwright(urls, schema=selected_schema)
    st.write("Extracted Content: ")
    st.write(extracted_content)

    # Determine the next available filename
    filename = get_next_filename()

    # Write the extracted content to the JSON file
    with open(filename, "w") as file:
        json.dump(extracted_content, file)

    st.write(f"Extracted content has been saved to {filename}")