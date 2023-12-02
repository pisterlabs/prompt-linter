# Import required libraries
from langchain.document_loaders import (  
    PyMuPDFLoader, # For loading PDF files
    DirectoryLoader, # For loading files from a directory 
    TextLoader, # For loading plain text files
    Docx2txtLoader, # For loading DOCX files
    UnstructuredPowerPointLoader, # For loading PPTX files 
    UnstructuredExcelLoader # For loading XLSX files
    
)
from langchain.document_loaders.csv_loader import CSVLoader # For loading CSV files  
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter # For splitting text into smaller chunks
from utils import (update_vectorstore_FAISS, update_vectorstore_PINECONE, convert_filename_to_key) 
 
from dotenv import load_dotenv # For loading environment variables from .env file

load_dotenv()
 
# Replace with the name of the directory carrying your data  
data_directory = "E:\\DESKTOP\\FreeLanceProjects\\muhammad_thaqib\\LawyerAI\\data"

# Load your documents from different sources
def get_documents():

    # Create loaders for PDF, text, CSV, DOCX, PPTX, XLSX files in the specified directory
    pdf_loader = DirectoryLoader(f"./{data_directory}", glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    txt_loader = DirectoryLoader(f"./{data_directory}", glob="**/*.txt", loader_cls=TextLoader)
    csv_loader = DirectoryLoader(f"./{data_directory}", glob="**/*.csv", loader_cls=CSVLoader) 
    docx_loader = DirectoryLoader(f"./{data_directory}", glob="**/*.docx", loader_cls=Docx2txtLoader) 
    pptx_loader = DirectoryLoader(f"./{data_directory}", glob="**/*.pptx", loader_cls=UnstructuredPowerPointLoader)
    xlsx_loader = DirectoryLoader(f"./{data_directory}", glob="**/*.xlsx", loader_cls=UnstructuredExcelLoader)
    
    # Initialize documents variable  
    docs = None

    # Load files using the respective loaders
    pdf_data = pdf_loader.load() 
    txt_data = txt_loader.load()
    csv_data = csv_loader.load()
    docx_data = docx_loader.load()
    pptx_data = pptx_loader.load() 
    xlsx_data = xlsx_loader.load()

    # Combine all loaded data into a single list  
    docs = pdf_data + txt_data + csv_data + docx_data + pptx_data + xlsx_data 

    # Return all loaded data
    return docs

# Get the raw documents from different sources
raw_docs = get_documents()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2300, chunk_overlap=10)

docs = text_splitter.split_documents(raw_docs)

# Print the number of documents and characters in the first document
print(f'You have {len(docs)} document(s) in your data')
print(f'There are {len(docs[0].page_content)} characters in your first document')

for object in docs:
    try:
        object.metadata["filename_key"] = convert_filename_to_key(os.path.split(object.metadata['source'])[-1])
    except Exception as oops:
        print(f"Object causing error is: {object}")


update_vectorstore_PINECONE(docs=docs)

""" # Start the interactive loop to take user queries
while True:
    query = input("Ask your query: ") # Take user input
    
    # Perform similarity search in the vector database and get the most similar documents
    docs = db.similarity_search(query, k=3)
    
    ans = utils.get_answer(query=query,context=docs)
    print(f"Answer: {ans}") """