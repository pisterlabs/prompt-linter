from flask import Flask,render_template,url_for, request,flash,redirect,session
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import sys
from backend.Uploader.parser import ResumeParser
import os
import backend.openapi as aapi
import openai
from backend.DataParser.fetchtodb import Fetcher


aapi.__init__()
fetcher = Fetcher()

UPLOAD_FOLDER = 'Resumes'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")

def index(name=None):
    url_for('static', filename='grayscale.js')
    return render_template("index.html",name=name)


@app.route("/FindJobs")
def findjobs():
    return render_template("Find Jobs.html")

@app.route("/FindEmployee")
def findemployees():
    return render_template("Find employee.html")

@app.route("/Mailed")
def mailed():
    return render_template("Mailed.html")

@app.route("/Candidates", methods = ['GET', 'POST'])    
def candidates():
    # if request.method == 'POST':
    #     data = request.form
    #     needed_position=data.get('position')
    #     needed_skills=data.get('skills')
    #     # search in database for id of that skill in skill table
    #     skill = fetcher.get_skill(needed_position)
    #     skill_id = skill[0]
        
    #     employee_id = fetcher.get_employee_id(skill_id)[0][0]
    #     employee_skill_id = fetcher.get_employee_id(skill_id)[1][0]
        
        
    #     candidate = fetcher.get_top_skills_by_rating(employee_skill_id)
    #     # search in employee_skill table for employee_id with that skill_id
    #     employee = fetcher.get_employee(employee_id)
    #     phone=employee[3]



        return render_template("Candidates.html")

@app.route("/uploader", methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            parsedCVText = ResumeParser(file.filename)
            parsedCV = parsedCVText.get_info()

            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """I will give you a resume of someone in the following format
                Full Name: [name]
                Email: [email]
                Phone: [phone]
                Address: [address]
                [category]: [score in percentage from 0.0 to 1.0]
                your job is to respond with only a comma seperated output for only the skills, so you only seperate the skills with a comma and no spaces between them. and if there are spelling mistakes fix those as well.
                """},
                {"role": "user", "content": parsedCV}])
            skill_list = completion.choices[0].message.content.strip().split(',')
            Fetcher.add_skill(skill_list)

            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """I will give you a resume of someone in the following format
                Full Name: [name]
                Email: [email]
                Phone: [phone]
                Address: [address]
                [category]: [score in percentage from 0.0 to 1.0]
                your job is to put Full Name, Email, phone and address into the following format (name,email,phone) without the parantheses
                """},
                {"role": "user", "content": parsedCV}])
            employee_info = completion.choices[0].message.content.strip().split(',')
            # Fetcher.add_employee(employee_info)
            Fetcher.add_rest(parsedCV,skill_list,employee_info)
            


            # Fetcher.get_skills()


            return render_template("/Mailed")

if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.debug = True
    
    app.run(host='0.0.0.0',port=5000)

