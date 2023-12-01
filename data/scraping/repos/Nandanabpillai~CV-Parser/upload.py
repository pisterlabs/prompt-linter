from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS  # Import CORS
from tempfile import NamedTemporaryFile
import pypdfium2 as pdfium
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain   
from PIL import Image
from datetime import datetime
import json
import base64
import pandas as pd
import ast
import nltk

os.environ["OPENAI_API_KEY"] = "sk-***************************"
app = Flask(__name__)
CORS(app)  # Add this line to enable CORS for your app

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Ensure the 'uploads' directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        with NamedTemporaryFile(delete=False, dir='.', suffix='.pdf') as f:
            f.write(file.getbuffer())
            PDFFileName = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pdf = pdfium.PdfDocument(PDFFileName)
            n_pages = len(pdf)
            for page_number in range(n_pages):
                page = pdf.get_page(page_number)
                pil_image = page.render(scale=4).to_pil()
        loader = UnstructuredPDFLoader(PDFFileName)
    pages = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(pages, embeddings).as_retriever()    
    
    current_date = datetime.now()
    query = "Output informatio, (all in English), from the document in JSON format: full name, contacts, age, languages, education, school, places of work, skills.If some fields cannot be filled from the document, then create this field and fill it with N/A. If the date of birth is not indicated, then please calculate the approximate age of the candidate based on the information provided in the document, for calculations, take into account that graduation from the university is usually at 22 years old. Current date = "+ current_date.date().strftime('%Y-%m-%d')
    docs = docsearch.get_relevant_documents(query)
    chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
    output = chain.run(input_documents=docs, question=query)
    valid_json = ast.literal_eval(output)
    
    json_data = json.loads(json.dumps(valid_json))

    names = [json_data.get("full_name", "N/A")]
    contacts = [json_data.get("contacts", "N/A")]
    ages = [json_data.get("age", "N/A")]
    languages = [json_data.get("languages", "N/A")]
    education = [json_data.get("education", "N/A")]
    school = [json_data.get("school", "N/A")]
    works = [json_data.get("places_of_work", "N/A")]
    skills = [json_data.get("skills", "N/A")]

    df = pd.DataFrame({
        "name": names,
        "contacts": contacts,
        "age": ages,
        "languages": languages,
        "education": education,
        "school": school,
        "work": works,
        "skill": skills
    })    
    csv = df.to_csv(index=False).encode('utf-8')
    

if __name__ == '__main__':
    app.run(host='localhost', port=8000)
