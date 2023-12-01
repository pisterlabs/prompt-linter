from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter


def get_pdf_text(pdf_document):
    """
    This function uses the PdfReader class in order to extract the text
    from the pdfs given as input.

    :param pdf_document: This takes in a pdf document to be extracted.
    :return: The output of this function is extracted raw text.
    """

    text = ""
    for pdf in pdf_document:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(raw_text):
    """
    This function uses the CharacterTextSplitter class from LangChain.text_splitter
    in order to convert the raw text outputted from the previous function into
    chunks.

    :param raw_text: The raw text to be converted into chunks.
    :return: The output of this function is the text being split into chunks, with
    chunk size of 1000 and chunk overlap of size 200.
    """

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks
