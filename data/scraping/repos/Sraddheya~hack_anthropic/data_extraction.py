# Files are all local to machine. This must be ammended.

# pip install PyPDF2
# pip install anthropic

import PyPDF2

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os

# app = Flask(__name__)
# CORS(app)

# @app.route('/uploadpdf', methods=['POST'])
# def upload_pdf():
#     if 'pdfFile' in request.files:
#         pdf_file = request.files['pdfFile']
#         # Process the PDF file here, e.g., save it, perform operations, etc.
#         # Example: pdf_file.save('uploaded_file.pdf')
#         print(pdf_file)
#         return 'PDF file uploaded successfully'
#     else:
#         return 'No PDF file received', 400

# if __name__ == '__main__':
#     app.run()

#create file object variable
#opening method will be rb
pdffileobj=open('test_cv.pdf','rb')

#create reader variable that will read the pdffileobj
pdfreader=PyPDF2.PdfReader(pdffileobj)

#This will store the number of pages of this pdf file
x=len(pdfreader.pages)
print("NUMBER OF PAGES " + str(x))

#create a variable that will select the selected number of pages
pageobj=pdfreader.pages[x - 1]

#(x+1) because python indentation starts with 0.
#create text variable which will store all text datafrom pdf file
text=pageobj.extract_text()


from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

output_format = """{
    "name": xxxx,
    "experience": xxxx}"""

job_format = """{
    "job_title": xxxx,
    "job_duration":xxxx,
    "company": xxxx,
    "skills": xxxx}"""

# Name
anthropic = Anthropic(api_key="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")
completion = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=300,
    prompt=f"{HUMAN_PROMPT} You will be extracting the most useful information from a resume. Extract the following information and present it in the JSON format: {output_format} and repeat this for every job listed in the JSON format {job_format}. Do not provide any preamble or closing, just the raw JSON. Extract the tools listed in each job description by their mentions. Round the job durations to their nearest whole month, for example if someone has been in a role from September 2021 to September 2021 this will count as 1 month and November 2021 to April 2022 will count as 6 months. <resume>{text}<resume> {AI_PROMPT}",
)
result = completion.completion
