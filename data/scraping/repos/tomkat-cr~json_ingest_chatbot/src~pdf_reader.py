from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


def get_pdf_pages(pdf_docs):
    all_pages = []
    for pdf in pdf_docs:
        from PyPDF2 import PdfReader, PdfWriter
        pdf_reader = PdfReader(pdf)
        pdf_writer = PdfWriter()

        for page in pdf_reader.pages:
            pdf_writer.add_page(page)

        with open(pdf.name, 'wb') as output_file:
            pdf_writer.write(output_file)
        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        loader = PyPDFLoader(pdf.name)
        pdf_pages = loader.load_and_split(text_splitter=text_splitter)
        all_pages += pdf_pages
    return all_pages
