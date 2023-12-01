from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
import pypdf
from langchain.schema import Document
import re
from random import choice


def process_pdf(folder, filename):
    print(f"Processing {filename}...")
    pdf_file = open(os.path.join(folder, filename), 'rb')
    reader = pypdf.PdfReader(pdf_file)
    number_of_pages = len(reader.pages)
    for page in range(1, number_of_pages - 3):
        page_content = reader.pages[page].extract_text()
        if page_content is None:
            continue
        if "BLANK PAGE" in page_content:
            continue

        page_content = re.sub(
            r"\d{0,2}\n(Turn\s+?over)?\s?© OCR 20\d\d", "", page_content)
        page_content = re.sub(r"END OF QUESTION PAPER", "", page_content)
        page_content = re.sub(r"PhysicsAndMathsTutor.com", "", page_content)

        page_content = re.sub(r"\.{10,}\n", "."*20, page_content)
        page_content = re.sub(r"\.{10,}( |\n)?", "."*20, page_content)
        page_content = re.sub(r"\.{10,}", "."*20, page_content)
        yield Document(
            page_content=page_content,
            metadata={
                "source": filename,
                "page": page+1
            }
        )
    pdf_file.close()


def process_folder(folder):
    pdf_dir = os.getcwd()
    for filename in os.listdir(folder):
        if filename.endswith('.pdf'):
            yield from process_pdf(folder, filename)


data = list(process_folder("./pdfs/chem-ocr"))
print("pages", len(data))
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=364, chunk_overlap=64)
all_splits = text_splitter.split_documents(data)
print("split chunks", len(all_splits))


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("embedding model loaded", embedding.model_name)

vectorstore = Chroma(persist_directory="./chroma_db/chem-combine",
                     embedding_function=embedding)

vectorstore.add_documents(all_splits)
