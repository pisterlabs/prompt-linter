from flask import Blueprint, request, jsonify
from .util.Embed import call_embeddings
import pinecone
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from src.database.TemplateInfo import JD
import fitz

upload = Blueprint('upload',__name__,url_prefix="/upload")

@upload.post('/')
def upload_file():
    global file_name
    
    file_object = request.files.get('file')
    file_name = file_object.filename

    print(file_name)

    print('============')
    doc = fitz.open(stream=file_object.read(), filetype="pdf")
    print('============')

    num_pages = len(doc)

    embeddings = OpenAIEmbeddings()

    pinecone.init(
        api_key="035dee03-042d-4a6e-bc1b-a9a4a3546f2b", 
        environment="northamerica-northeast1-gcp",  
    )
    index_name = "langchain1"

    pdf_texts = []

    for page_num in range(num_pages):
        page = doc[page_num]
        text = page.get_text()

        pdf_texts.append(text)

       
    print(len(pdf_texts))

    vectorstore = Pinecone.from_texts(pdf_texts, embeddings, index_name=index_name, namespace=file_name)
    JD[1]  = file_name

    print(file_object)
    
    return jsonify({"POST":"test"})
    
    
    

