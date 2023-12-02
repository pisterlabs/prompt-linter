# Python program that reads files from a folder, creates embeddings using OpenAI, stores them in a Chromata vectorstore, 
# and responds to user questions using OpenAI LLM with prompt engineering to ensure correct answers:
import os
import re
from docx import Document
from langchain import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import PGEmbedding
import pdfminer.high_level
import psycopg2

# Clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')

## Set local environment variables
folder_path = "QnA_Examples/content"
OPENAI_API_KEY=os.getenv("OPEN_API_KEY")
db_user=os.getenv("DBUSER")
db_password=os.getenv("DBPASSWORD")
db_host=os.getenv("DBHOST")

embeddings = OpenAIEmbeddings()

## Set text splitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 500,
    chunk_overlap  = 50
)
embedding = OpenAIEmbeddings(request_timeout=60)
# embeddingmodel="text-embedding-ada-002"
files = os.listdir(folder_path)

## Convert pdf to text
def convert_pdf_to_text(full_path):
    with open(full_path, 'rb') as f:
        output = pdfminer.high_level.extract_text(f)
    return output

## Process pdf document
def process_pdf_document(full_path):
    print("process_pdf_document: " + full_path)
    text = convert_pdf_to_text(full_path)
    # Cleanse text
    text = cleanse_text(text)
    # Create chunks
    chunks = chunk_text(text)
    # Create embeddings
    create_embeddings(chunks)

## Process docx document
def process_docx_document(full_path):
    print("process_docx_document: " + full_path)
    doc = Document(full_path)
     # Extract text
    text = '\n'.join([para.text for para in doc.paragraphs])
    # Cleanse text
    text = cleanse_text(text)
    # print("text: " + text)
    # Create chunks
    chunks = chunk_text(text)
    # Create embeddings
    create_embeddings(chunks)


## Process text document
def process_txt_document(full_path):
    print("process_txt_document: " + full_path)
    # Extract text
    with open(full_path, 'r') as file:
        text = file.read()
    # Cleanse text
    text = cleanse_text(text)
    #print("text: " + text)
    # Create chunks
    chunks = chunk_text(text)
    # Create embeddings
    create_embeddings(chunks)

## Chunk text
def chunk_text(text):
    print("chunk_text")
    chunks = []
    texts = text_splitter.create_documents([text])
    #print("Number of records in texts: " + str(len(texts)))
    for i, text in enumerate(texts):
        chunk_string = str(texts[i].page_content)
        #print("chunk number: " + str(i) + " chunk string: " +  chunk_string)
        chunks.append(chunk_string)
    return chunks

## Cleanse text
def cleanse_text(text):
    print("cleanse_text")
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Convert all characters to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    # Replace \n with a space
    text = text.replace('\n', '')
    return text

## Create embeddings
def create_embeddings(chunks):
    # Connect to the PostgreSQL database server
    conn = psycopg2.connect(
        dbname="llm-demo",
        user=db_user,
        password=db_password,
        host=db_host,
        port="5432"
        )
   
    # Truncate the coneptembedding table
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE contentembedding")
    conn.commit()

    # Open a cursor to perform database operations
    cur = conn.cursor()

    embedding_vectors = []
    for i, chunk in enumerate(chunks):      
        if chunks[i] != ' ': # Skip empty chunks
            embedding_content = str(chunks[i])
            embedding_vector = embeddings.embed_query(chunks[i])
            cur.execute("INSERT INTO contentembedding (embedding_content, embedding_vector) VALUES ( %s, %s)", (embedding_content, embedding_vector))
   
    # Commit changes
    conn.commit()
    # Close cursor and connection
    cur.close()
    conn.close()

## Process all files in the folder
for file in files:
    if file.endswith('.pdf'):
        full_path = os.path.join(folder_path, file)
        #print(" In pdf: " + full_path)
        process_pdf_document(full_path)
        # document = PyPDFLoader(os.path.join(folder_path, file))
    if file.endswith('.docx'):  
        full_path = os.path.join(folder_path, file)
        print(" In docx: " + full_path)
        process_docx_document(full_path)
    elif file.endswith('.txt'):
        full_path = os.path.join(folder_path, file)
        print(" In txt: " + full_path)
        process_txt_document(full_path)
      