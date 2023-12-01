from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

def extract_information(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        pdf = PdfReader(f)
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

def export_vector_store(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings()
    db = FAISS.from_texts(text_chunks, embedding=embeddings)
    db.save_local('Arch_index')
    
    
    

pdffile = './Docs/arch.pdf'

raw_text = extract_information(pdffile)

text_chunks = get_text_chunks(raw_text)

export_vector_store(text_chunks=text_chunks)







