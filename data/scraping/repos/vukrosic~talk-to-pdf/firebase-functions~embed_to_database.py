import io
import os
import pinecone
import openai
from PyPDF2 import PdfReader
from firebase_functions import https_fn
from firebase_functions import options
from firebase_admin import initialize_app

initialize_app()
options.set_global_options(max_instances=10)

@https_fn.on_request(
    cors=options.CorsOptions(
        cors_origins=["http://localhost:3000", "*"],
        cors_methods=["GET", "POST"],
    )
)
def initialize_pinecone():
    PINECONE_API_KEY = ""
    PINECONE_API_ENV = ""
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    print("Pinecone initialized")
    return pinecone

def embed_to_database(req: https_fn.Request) -> https_fn.Response:
    


    # Convert the Request object to a dictionary
    try:
        # Check if a file was uploaded
        if "file" in req.files:
            uploaded_file = req.files["file"]
            file_extension = uploaded_file.filename.split(".")[-1].lower()
            print(f"File extension: {file_extension}")

            # Check if the uploaded file is a PDF
            if file_extension == "pdf":
                # Extract text from the PDF file
                file_content = uploaded_file.read()
                pdf_text = extract_text_from_pdf_file(file_content)
                print("PDF text extracted")
            else:
                pdf_text = ""
                print("No PDF file uploaded")
       

        # Check if user-pasted text is provided
        user_pasted_text = req.form.get('userPastedText', '')  # Access userPastedText from req.form
        print("User-pasted text:")
        print(user_pasted_text)

        # Combine text from PDF and user-pasted text
        combined_text = pdf_text + user_pasted_text

        # Convert the extracted text to OpenAI vector embeddings
        chunk_size = 1000
        texts = [combined_text[i:i + chunk_size] for i in range(0, len(combined_text), chunk_size)]
        index_name = "test"  # Replace with your actual Pinecone index name
        pinecone.init(api_key="", environment="")
        print("Pinecone initialized for text embedding")
        
        # Upload the extracted text to Pinecone
        upload_data_to_pinecone(texts, index_name, pinecone)
        print("Data uploaded to Pinecone")

        # Return a response
        return https_fn.Response("Text extracted and uploaded to Pinecone successfully")
    except Exception as e:
        error_message = f"Error: {str(e)}"
        print(error_message)
        return https_fn.Response(error_message, status=500)

def extract_text_from_pdf_file(pdf_bytes):
    text = ""
    # Debugging: Print the length of the PDF bytes
    pdf_bytes_length = len(pdf_bytes)
    print(f"PDF bytes length: {pdf_bytes_length}")
    
    pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def upload_data_to_pinecone(texts, index_name, pinecone):
    # Initialize OpenAI API client
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    print(os.environ.get("OPENAI_API_KEY"))
    # Convert and upload data as tuples (ID, vector)
    data_to_upload = [(str(i), embed_text(text), {"text": text}) for i, text in enumerate(texts)]
    print(f"Data to upload: {data_to_upload}")
    
    # Upload the data to Pinecone
    index = pinecone.Index("test")
    index.delete(delete_all=True)
    index.upsert(data_to_upload)
    print("Data uploaded to Pinecone")

def embed_text(text):
    print("text: ")
    print(text)
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']
    print("Text embedded")
    return embeddings
