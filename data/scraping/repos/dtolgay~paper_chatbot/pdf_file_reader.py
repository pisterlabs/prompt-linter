# def read_pdf(pdf_file_path):
#     from pdfminer.high_level import extract_text
#     text = extract_text(pdf_file_path)
#     return text


# def save_embeddings_into_csv_file():
#     import csv 
#     import PyPDF2


def read_PDF(name_of_the_paper, extension='pdf'):
    import langchain
    pdf_file_path = name_of_the_paper + '.' + extension
    loader = langchain.document_loaders.PyPDFLoader(pdf_file_path)
    pages = loader.load_and_split()    
    return pages
