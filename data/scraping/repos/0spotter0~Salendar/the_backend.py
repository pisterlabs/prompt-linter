from flask import Flask, request, jsonify
import fitz
from werkzeug.utils import secure_filename
import os
import openai
import docx2pdf
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': 'http://localhost:3000'}})

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/upload', methods=['POST'])
def upload_file():
    print("in backend upload")

    print(f"request: {request}")
    print(f"request something: {request.url}")

    if 'file' not in request.files:
        print("file not in request.files")  
        return jsonify({'error': 'No file part'})

    print("about to get file")

    print(f"request.files: {request.files}")

    file = request.files['file']
    
    print(f"file: {file}")

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename, {'txt', 'pdf', 'docx'}):
        filename = secure_filename(file.filename)
        filepath = os.path.join("./uploaded_files", filename)
        file.save(filepath)

        print(f"saved file to {filepath}")

        if filepath.endswith('docx'):
            docx2pdf.convert(filepath, f'{filepath}.pdf')
            print(f"removing :) file: {filepath}")
            os.remove(filepath)
            filepath = f'{filepath}.pdf'
            
        syllabus = fitz.open(filepath)

        print(f"removing :( file: {filepath}")
        os.remove(filepath)

        raw_text = ""
        for page in syllabus:
            text = page.get_text()
            raw_text += text

        output = getGPT(raw_text)
        
        response = jsonify({'syllabus_text': output})
        response.status_code = 200

        return response
    else:
        return jsonify({'Error': 'File type not allowed'})



base_prompt = """
This is a syllabus. Ignore the course code and title. Tell me all the dates of each assignment and due date in the same JSON structure as the example below:

{
    "title": "Psychology 101",
    "assignments": [
        {
            "title": "Homework 1",
            "due_day": 14,
            "due_month": 4,
            "due_year": 2023
        }
    ]
}
"""

def getGPT(syllabus_text):
    print(f"in get gpt with text {syllabus_text}")
    openai.api_key = os.getenv("OPENAI_API_KEY")

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": "You are a helpful assistant."},
            {"role": "system", "content": base_prompt + syllabus_text},
            ]
        )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)