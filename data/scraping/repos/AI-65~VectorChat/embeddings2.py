import os
import logging
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
import re

# This sets up logging so you can see what the script is doing and track any issues that might come up.
logging.basicConfig(filename='logfilembedding.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# This function is used to remove any characters from a string that aren't allowed in filenames.
def make_filename_safe(filename):
    return re.sub(r'[<>:"/\\|?*§]', '', filename)  # Added '§' to the list of characters to be replaced

# This function takes in a path to a PDF and extracts all the text from it.
def get_pdf_text(pdf):
    logging.info(f"Extracting text from {pdf}...")
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    logging.info(f"Finished extracting text from {pdf}.")
    return text

# This function takes in a long string of text and splits it into smaller chunks.
def get_text_chunks(text):
    logging.info("Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    logging.info("Finished splitting text into chunks.")
    return chunks

# This function takes in a list of text chunks, and creates a FAISS vectorstore from it. It then saves this vectorstore to disk.
def get_vectorstore(text_chunks, save_path='faiss_index', load_if_exists=True):
    logging.info("Creating vectorstore...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    logging.info("Finished creating vectorstore. Saving to disk...")
    vectorstore.save_local(save_path)
    logging.info("Saved vectorstore to disk.")

# This is the main function, which uses all the previous functions to perform the overall process.
def main():
    load_dotenv()

    # List all PDF files in the project folder
    pdf_docs = [f for f in os.listdir(".") if f.endswith(".pdf")]

    # Extract text from each PDF and create a vectorstore
    for pdf in pdf_docs:
        # Extract text
        text = get_pdf_text(pdf)

        # Split the text into chunks
        text_chunks = get_text_chunks(text)

        # Start tracking token usage (not working rn)
        with get_openai_callback() as cb:
            # Create a vectorstore from these text chunks
            # The save_path is set to the name of the PDF, replacing ".pdf" with "_vectorstore"
            safe_filename = make_filename_safe(pdf.replace('.pdf', ''))
            get_vectorstore(text_chunks, save_path=f"{safe_filename}_vectorstore")

            logging.info(f"Total tokens used for {pdf}: {cb.total_tokens}")
            logging.info(f"Total cost (USD) for {pdf}: ${cb.total_cost}")
            logging.info(f"Prompt Tokens for {pdf}: {cb.prompt_tokens}")
            logging.info(f"Completion Tokens for {pdf}: {cb.completion_tokens}")

if __name__ == "__main__":
    main()
