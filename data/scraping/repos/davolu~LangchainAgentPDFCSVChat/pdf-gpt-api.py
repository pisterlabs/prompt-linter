import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

load_dotenv()
# Initialize Flask app 
app = Flask(__name__)
# Create a directory for storing uploaded files within the app context
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
@app.route('/process_pdf', methods=['POST'])
def process_pdf():


    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        return jsonify({"error": "OPENAI_API_KEY is not set"}), 500

    pdf_file = request.files.get('pdf_file')
    if pdf_file is None:
        return jsonify({"error": "No PDF file provided"}), 400

    # Get the original file name
    original_filename = pdf_file.filename
    
    # Create a path for saving the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, original_filename)

    # Save the uploaded file with the original name
    pdf_file.save(file_path)

    # location of the pdf file/files. 
    reader = PdfReader(file_path)

    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    query = "I'm 28 years old. Can I run for presidency?"
    docs = docsearch.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)

    
    # You can format the response as needed, e.g., convert to JSON
    response_json = {"answer": response}
    
    return jsonify(response_json), 200

if __name__ == "__main__":
    app.run(debug=True)
