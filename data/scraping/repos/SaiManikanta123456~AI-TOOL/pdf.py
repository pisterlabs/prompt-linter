import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()

st.title("PDF Question Answering")

# Sidebar for uploading PDFs
st.sidebar.title("Upload PDFs")
pdf_files = st.sidebar.file_uploader("Choose up to 3 PDF files", accept_multiple_files=True, type="pdf")

# Function to process PDFs and display answer
def process_pdfs_and_answer(pdf_files, question):
    raw_text = ''

    # Read text from PDFs
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

    # Split text into chunks and create embeddings
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)

    # Load question answering chain and run the question
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    docs = document_search.similarity_search(question)
    result = chain.run(input_documents=docs, question=question)

    return result

# Main content
if pdf_files:
    # Allow the user to input a question
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        # Process PDFs and display the answer
        result = process_pdfs_and_answer(pdf_files, question)
        st.write("### Answer:")
        st.write(result)
else:
    st.write("Please upload PDFs in the sidebar.")


