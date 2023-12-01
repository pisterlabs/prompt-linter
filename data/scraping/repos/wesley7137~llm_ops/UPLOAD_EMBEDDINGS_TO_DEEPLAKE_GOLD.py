from langchain.vectorstores import DeepLake
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import os
import json
import csv
from PyPDF2 import PdfReader

# Initialize environment variables and vector memory
from langchain.vectorstores import DeepLake
############################# ENVIRONMENT VARIABLES
print("Initializing API keys and environment variables...")
os.environ["ACTIVELOOP_TOKEN"] = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTY5NzQzMjMyNywiZXhwIjoxNzMwNDM3MTEzfQ.eyJpZCI6Indlc2xhZ2FyZGUifQ.N5ZYpozHB2tFDlFhVqyA-1ut-NGtaJF3WM-OG22AR0ispilW_M6kwzXnx4hHGMov3ESfhVw0jKCt-hfdsdqQnw"
os.environ["ACTIVELOOP_USERNAME"] = "weslagarde"
print("Initializing vector memory.")
dataset_path = "hub://weslagarde/PDF_DB_GOOD"
print("Vector memory initialized successfully.")
embeddings = HuggingFaceEmbeddings()
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

# Function to split text into chunks
def split_into_token_chunks(text, num_tokens=300):
    tokens = text.split()
    return [" ".join(tokens[i:i + num_tokens]) + "\\n\\n" for i in range(0, len(tokens), num_tokens)]

# Initialize an empty list to hold the documents
documents = []

# Prompt user for file or directory path
path = input("Enter the file or directory path: ")
print(f"Reading files from {path}...")

# Check if the path is a directory
if os.path.isdir(path):
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        
        # Read and chunk TXT files
        if filename.endswith(".txt"):
            with open(filepath, 'r') as f:
                text = f.read()
            chunks = split_into_token_chunks(text)
            documents.extend(chunks)
        
        # Read and chunk JSON files
        elif filename.endswith(".json"):
            with open(filepath, 'r') as f:
                json_objects = json.load(f)
            for obj in json_objects:
                documents.append(json.dumps(obj))
        
        # Read and keep Python files intact
        elif filename.endswith(".py"):
            with open(filepath, 'r') as f:
                py_content = f.read()
            documents.append(py_content)
        
        # Read and chunk CSV files
        elif filename.endswith(".csv"):
            with open(filepath, 'r') as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    documents.append(",".join(row))

        
        # Read and chunk PDF files
        elif filename.endswith(".pdf"):
            pdf_reader = PdfReader(open(filepath, "rb"))
            pdf_text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                pdf_text += page.extract_text()
            chunks = split_into_token_chunks(pdf_text)
            documents.extend(chunks)





# Read and chunk TXT files
if path.endswith(".txt"):
    with open(path, 'r') as f:
        text = f.read()
    chunks = split_into_token_chunks(text)
    documents.extend(chunks)

# Read and chunk JSON files
if path.endswith(".json"):
    with open(path, 'r') as f:
        json_objects = json.load(f)
    for obj in json_objects:
        documents.append(json.dumps(obj))

# Read and keep Python files intact
if path.endswith(".py"):
    with open(path, 'r') as f:
        py_content = f.read()
    documents.append(py_content)

# Read and chunk CSV files
if path.endswith(".csv"):
    with open(path, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            documents.append(",".join(row))


# Read and chunk PDF files
if path.endswith(".pdf"):
    pdf_reader = PdfReader(open(path, "rb"))
    pdf_text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()
    chunks = split_into_token_chunks(pdf_text)
    documents.extend(chunks)

# Upload to DeepLake
if documents:
    print(f"Uploading {len(documents)} documents to DeepLake...")
    
    # Print the first 5 examples of what's being uploaded
    print("First 5 examples of documents being uploaded:")
    for example in documents[:15]:
        print(example)
        
    db.add_texts(texts=documents)
    print("Documents uploaded to database successfully.")
else:
    print("No documents to upload.")
