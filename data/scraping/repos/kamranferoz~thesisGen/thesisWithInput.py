import streamlit as st
import fitz  # PyMuPDF
import re
import openai

# Set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit UI components
st.title("Thesis Generator App")

# User inputs
title = st.text_input("Title:", "Machine Learning Algorithms")
user_name = st.text_input("Name:", "Your Name")

st.write(f"Title: {title}")
st.write(f"Name: {user_name}")

st.write("Upload multiple PDF files and generate a thesis with citations.")

# Upload multiple PDF files
pdf_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        text += pdf_document.get_page_text(page_num)
    return text

# Function to add citations to text
def add_citations_to_text(text, citations):
    # Replace [1], [2], ... with actual citations
    citation_pattern = re.compile(r'\[(\d+)\]')

    def replace_citations(match):
        reference_number = match.group(1)
        # Replace with actual citation from citations list
        if citations and int(reference_number) <= len(citations):
            return citations[int(reference_number) - 1]
        return f"[{reference_number}]"

    processed_text = citation_pattern.sub(replace_citations, text)
    return processed_text

# Generate thesis button
if st.button("Generate Thesis"):
    if pdf_files:
        # Extract citations from user input
        citations = st.text_area("Citations (one per line)", "")
        citations = citations.strip().split('\n')

        # Generate thesis using OpenAI's GPT-3 model
        prompt = f"Title: {title}\nName: {user_name}\nTask: Generate a thesis on the topic of {title}.\n"
        for idx, citation in enumerate(citations, start=1):
            prompt += f"Citation {idx}: {citation}\n"
        response = openai.Completion.create(
            engine="text-davinci-003",  # Choose appropriate engine
            prompt=prompt,
            max_tokens=800,  # Adjust as needed
        )

        generated_thesis = response.choices[0].text.strip()

        # Add citations to generated thesis
        thesis_with_citations = add_citations_to_text(generated_thesis, citations)

        # Display the generated thesis with citations
        st.write(f"Generated Thesis for {title} with Citations:")
        st.write(thesis_with_citations)
    else:
        st.warning("Please upload at least one PDF file.")
 
# Streamlit app footer
st.write(f"Created by {user_name}")
