import os
from flask import Flask, render_template, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request
import cohere
import openai
from pdfminer.high_level import extract_text
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename

# from dotenv import load_dotenv


# load_dotenv()


# COHERE = os.environ.get('COHERE_KEY')
COHERE = os.getenv('COHERE_KEY')

# co = cohere.Client(COHERE)


openai.api_key = os.getenv("OPEN_AI_KEY")
UPLOAD_FOLDER = './files'
app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
# app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')

# db = SQLAlchemy(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello_world():
    result = "this is so cool"
    return render_template('index.html', result = result)



@app.route('/get_data', methods=[ 'POST'])
def get_data():    
    # transcribe data
    uploaded_file = request.files['operation']
    
    filename = secure_filename(uploaded_file.filename)
    uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    # response = co.summarize(text=KNOWLEDGE)
    # print(response.summary)


    return 'success'

@app.route('/files/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


@app.route("/handle_question", methods = ['GET', 'POST'])
def handle_question():
    KNOWLEDGE = ''
    question = request.form['emp_question']

    directory = './files'
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            uploaded_file = download_file(filename)
            reader = PdfReader('./files/' + filename)
            number_of_pages = len(reader.pages)
            for page in reader.pages:
                KNOWLEDGE += page.extract_text()                
            # page = reader.pages[0]
            # KNOWLEDGE = page.extract_text()
            

    
    
    completion = openai.ChatCompletion.create(
     model= "gpt-3.5-turbo",
     messages= [{"role": "assistant", "content": "Based on this text, " + KNOWLEDGE + ", answer this user's question: " + question}],
     temperature= 0.7
    )

    

    return completion.choices[0].message.content




# if __name__ == '__main__' :
#     db.create_all()
#     app.run(debug = True )