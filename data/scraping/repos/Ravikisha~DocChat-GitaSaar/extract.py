import re
import json
import PyPDF2
import io
from langchain.document_loaders import PyPDFLoader

# from langchain_faiss import VectorDB

# Load the PDF file and extract the text
with open("gita.pdf", "rb") as f:
    pdf_data = f.read()
    # Use PyPDF2 to extract the text from the PDF file
    text = PyPDF2.PdfReader(io.BytesIO(pdf_data)).pages[0].extract_text()
    
loader = PyPDFLoader("./gita.pdf")
raw_documents = loader.load_and_split()
listToStr = ' '.join([str(elem) for elem in raw_documents])
print("Raw Document: ",listToStr)
# Use regular expressions to extract the data from the text
pattern = r"TEXT\s+(\d+)\s+([\s\S]*?)TRANSLATION\s+([\s\S]*?)PURPORT\s+([\s\S]*?)$"
match = re.search(pattern, listToStr)
if match:
    slog = match.group(2).strip()
    prons = match.group(3).strip()
    meaning = match.group(4).strip()
    translation = match.group(5).strip()
    purport = match.group(6).strip()

    # Create a dictionary to store the extracted data
    data = {
        "slog": slog,
        "prons": prons,
        "meaning": meaning,
        "translation": translation,
        "purport": purport,
    }

    # Store the data in a vector database using langchain FAISS
    # db = VectorDB("gita.db", data_dim=5)
    # db.add_item(data["slog"], data)
    
    print("Data : ",data)
