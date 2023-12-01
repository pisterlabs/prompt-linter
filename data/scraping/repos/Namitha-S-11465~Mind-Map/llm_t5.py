import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64

# Load T5 model and tokenizer
checkpoint = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(checkpoint)
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Function to preprocess file
def file_preprocessing(file):
    loader = PyPDFLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    return final_texts

# Function to generate summary
def generate_summary(filepath):
    input_text = file_preprocessing(filepath)
    summary = summarizer(input_text, max_length=150, min_length=30, num_beams=4, early_stopping=True)
    return summary[0]['summary_text']

# Function to display PDF
@st.cache_resource
def read_pdf(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    return base64_pdf

def display_pdf(file):
    base64_pdf = read_pdf(file)
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Document Summarization App using T5 Model")

uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

if uploaded_file is not None:
    if st.button("Summarize"):
        col1, col2 = st.columns(2)
        filepath = "data/" + uploaded_file.name
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        with col1:
            st.info("Uploaded File")
            read_pdf(filepath)  # Trigger the caching explicitly
            display_pdf(filepath)

        with col2:
            summary_text = generate_summary(filepath)
            st.info("Summarization Complete")
            st.success(summary_text)
