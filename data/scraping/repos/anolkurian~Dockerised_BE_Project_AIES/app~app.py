

from flask import Flask
import os
# from flask_pymongo import PyMongo
# from bson.json_util import dumps
# from bson.objectid import ObjectId
from flask import jsonify, request, render_template, url_for, session, redirect, flash, make_response
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from video import video
# from text import text
# from prosody import prosody
from video_audio import destroy, check_audio
from coherence import coherence_cv
from coherence import coherence_ans
from prosody import prosodyfile
# from text import textfile
from text import loadProfile
from text import getSummary
import collections
import functools
import operator
import ast
import operator
from collections import Counter
import cv2
# import pdfkit
# from flask_wkhtmltopdf import Wkhtmltopdf
import re
from pymongo import MongoClient
# Validation Stuff
# from flask_wtf import FlaskForm
# from wtforms import StringField, PasswordField, BooleanField, SubmitField
# from wtforms.validators import DataRequired

app = Flask(__name__)
app.register_blueprint(video, url_prefix="")
# wkhtmltopdf = Wkhtmltopdf(app)
# app.register_blueprint(text,url_prefix="")
# app.register_blueprint(prosody,url_prefix="")
# app.register_blueprint(coherence,url_prefix="")

UPLOAD_FOLDER = './uploads'

app.secret_key = "secretkey"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# app.config['MONGO_URI'] = "mongodb+srv://anol:anol@cluster0-1cvez.mongodb.net/test?retryWrites=true&w=majority"
# app.config['MONGO_URI'] = "mongodb://localhost:27017/interview_training"
# mongo = PyMongo(app)

mongo = MongoClient(
    'mongodb+srv://anol:anol@cluster0-1cvez.mongodb.net/test?retryWrites=true&w=majority')
db = mongo.get_database('interview_training')

# WKHTMLTOPDF_BIN_PATH = r'C:\Program Files\wkhtmltopdf\bin'
PDF_DIR_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'static', 'pdf')

# WKHTMLTOPDF_USE_CELERY = True

# class LoginForm(FlaskForm):
#     username = StringField('Username', validators=[DataRequired()])
#     password = PasswordField('Password', validators=[DataRequired()])
#     remember_me = BooleanField('Remember Me')
#     submit = SubmitField('Sign In')


def dumpsProfile(result_json):
    db.result.insert_one(result_json)


@app.route('/')
def index():
    finish()
    return render_template("home1.html")


@app.route('/student')
def student():
    prof = []
    profiles = {}
    for x in db.profile.find({}, {'_id': 0, 'comp_profile': 1}):
        for y in x.values():
            prof.append(y)
    print(prof)
    finish()
    return render_template("student.html", profiles=prof)


@app.route('/loginForm')
def loginForm():
    return render_template("login.html")


@app.route('/signupForm')
def signupForm():
    return render_template("signup.html")


@app.route('/studentPage')
def studentPage():
    return render_template("student2.html")


@app.route('/adminPage')
def adminPage():
    if 'user' in session:
        return render_template("admin.html")
    else:
        return render_template("login.html")


@app.route('/admin')
def admin():
    return render_template("admin.html")


@app.route('/submitCompany')
def submitCompany():
    return redirect(url_for("manageProfile"))


@app.route('/manageProfile')
def manageProfile():
    comp_profile = []
    description = []
    keywords = []
    questions = []
    for x in db.profile.find({}, {'_id': 0, 'comp_profile': 1}):
        for y in x.values():
            comp_profile.append(y)
    for x in db.profile.find({}, {'_id': 0, 'description': 1}):
        for y in x.values():
            description.append(y)
    for x in db.profile.find({}, {'_id': 0, 'keywords': 1}):
        for y in x.values():
            keywords.append(y)
    for x in db.profile.find({}, {'_id': 0, 'questions': 1}):
        for y in x.values():
            questions.append(y)
    print("q")
    print(questions)
    print("q")
    print(keywords)
    print("q")
    print(description)
    print("q")
    print(comp_profile)
    length = len(comp_profile)
    return render_template("manage.html", comp_profile=comp_profile, length=length,
                           questions=questions, description=description, keywords=keywords)


@app.route('/deleteProfile', methods=['POST'])
def deleteProfile():
    print("delete profile")
    values = request.form
    _value = values['comp_profile']
    print(_value)
    if _value and request.method == "POST":
        id = db.profile.find_one_and_delete(
            {"comp_profile": _value})
        print(id)

    return redirect(url_for("manageProfile"))


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('adminPage'))


# def allowed_file(filename):
#     return '.' in filename and \
#         filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/studentAccess', methods=['POST'])
def studentAccess():
    values = request.form
    _name = values['name']
    print(_name)
    _email = values['email']
    print(_email)
    _profile = values['comp-profile']
    print(_profile)
    session['stud_profile'] = _profile
    # if 'resume' in request.files:
    #     resume = request.files['resume']
    #     mongo.save_file(resume.filename, resume)
    #     filename = secure_filename(resume.filename)
    #     # filename = secure_filename(file.filename)
    #     # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    #     resume.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    if 'file' in request.files:
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    # if _name and _profile and _email and request.method == "POST":
    #     id = db.student.insert(
    #         {"name": _name, "email": _email, "comp_profile": _profile, "resume": resume.filename})
    if _name and _profile and _email and request.method == "POST":
        id = db.student.insert(
            {"name": _name, "email": _email, "comp_profile": _profile, })
    print("hello")
    if 'stud_profile' in session:
        stud = session['stud_profile']
        print(stud)
    keys = db.profile.find_one({'comp_profile': stud}, {
        '_id': 0, 'description': 0, 'comp_profile': 0, 'questions': 0})
    print(keys)
    for key in keys.values():
        session['keywords'] = key
    # session['keywords']=_keyword
    if 'keywords' in session:
        print(session['keywords'])
    questions = []

    value = db.profile.find_one({'comp_profile': stud}, {
        '_id': 0, 'description': 0, 'keywords': 0, 'comp_profile': 0})

    # print(value.values())
    # store all the questions in session
    for value in value.values():
        session['questions'] = value

    # add to the next button to display the next question
    if session['questions']:
        myques = session['questions']
        popped = myques.pop(0)
        session['questions'] = myques
        print(popped)

        return render_template('student2.html', question=popped)


@app.route('/adminAccess', methods=['POST'])
def adminAcess():
    values = request.form
    _username = values['username']
    _password = values['pass']
    session['user'] = _username
    print(session['user'])
    if _username and _password and request.method == "POST":
        valid = db.admin.find_one(
            {"username": _username, "pass": _password})
        if(valid):
            return redirect(url_for("adminPage"))
        else:
            flash('Incorrect Email Address or Password')
            return redirect(url_for('loginForm'))


@app.route('/signup', methods=['POST'])
def adminSignup():
    values = request.form
    _username = values['username']
    _email = values['email']
    _password = values['pass']
    present = db.admin.find_one({"username": _username})
    if present:
        flash('Email address already exists')
        return render_template("signup.html")
    else:
        if _username and _email and _password and request.method == "POST":
            id = db.admin.insert(
                {"username": _username, "email": _email, "password": _password})
            return render_template("login.html")


@app.route('/description', methods=['POST'])
def description():
    print("desc")
    values = request.form
    _value = values['profile']
    d = []
    desc = db.profile.find_one({'comp_profile': _value}, {
        '_id': 0, 'questions': 0, 'keywords': 0, 'comp_profile': 0})
    for c in desc.values():
        d.append(c)
    print(d)
    return jsonify(description=d)


@app.route('/companyDetails', methods=['POST'])
def companyDetails():
    print("companyDetails")
    values = request.form
    _profile = values['profile']
    _description = values[F'description']
    _keywords = values['keywords']
    _keyword = _keywords.split(',')

    session['comp_profile'] = _profile

    # session['description']=_description
    # session['keywords']=_keyword
    print(_profile)
    if _profile and _description and _keywords and request.method == "POST":
        id = db.profile.insert(
            {"comp_profile": _profile, "description": _description, "keywords": [_keyword]})
    return render_template("admin.html", profile=_profile, description=_description, keywords=_keywords)
    # return redirect(url_for("quest", profile=_profile, description=_description, keywords=_keywords))
    # return redirect(url_for('adminPage',profile=_profile, description=_description, keywords=_keywords))


@app.route('/addQuestion', methods=['POST'])
def addQuestion():
    print("add question")
    values = request.form
    _value = values['quest']
    print(_value)
    if 'user' in session:
        user = session['user']
    # keywords = values['keywords']
    # _keyword = keywords.split(',')
    # print(user)
    if 'comp_profile' in session:
        pro = session['comp_profile']
    print(pro)
    if _value and request.method == "POST":
        id = db.profile.find_one_and_update(
            {"comp_profile": pro}, {'$push': {"questions": _value}})
        return redirect(url_for("admin"))


@app.route('/deleteQuestion', methods=['POST'])
def deleteQuestion():
    print("delete question")
    values = request.form
    _value = values['quest']
    print(_value)
    if 'profile' in session:
        pro = session['profile']
        print(pro)
    if _value and request.method == "POST":
        id = db.profile.find_one_and_delete(
            {"comp_profile": pro}, {"questions": _value})
        return redirect(url_for("admin"))


@app.route('/afterloading')
def afterloading():
    session["garbage"] = "False"
    content = []
    print("afterloading")
    # prosody file
    finish_audio_files()
    prosodyfile()
    print("ass")
    with open("./uploads/audio_emotions.txt") as f:
        content = f.read()
    content = content.replace("trainingData/", "")
    content = content.split("\n")
    content = [x.strip() for x in content]
    content.pop()
    with open("./uploads/audio_coordinates.txt") as f:
        au = f.read()
    au = au.replace("[", "")
    au = au.replace("]", "")
    au = au.split("\n")
    au.pop()
    print(au)
    print(type(au))
    resultant = [0, 0, 0, 0, 0, 0]
    lenz = len(au)
    for x in au:
        # print(x)
        res = list(map(float, x.split()))
        # print("res")
        # print(res)
        # print(res[0])
        for y in range(5):
            # print(y)
            resultant[y] = resultant[y] + res[y]
            # print("resultant")
            # print(resultant)
    au = [r/lenz for r in resultant]
    print("au")
    lz=len(au)
    for i in range(lz):
        au[i]=round(au[i],4)
        print("au",au[i])
    print(au)
    print("after prosodyfile")
    print("before emotions")
    with open("./uploads/emotions.txt") as f:
        contente = f.readlines()
    contente = [x.strip() for x in contente]
    print("contente")
    print(contente)
    resultant = {}
    resultant = Counter(resultant)
    with open("./uploads/emotions_coordinates.txt") as f:
        cp = f.read()
    co = cp.split("\n")
    print(co)
    print(type(co))
    co.pop()
    for x in co:
        res = ast.literal_eval(x)
        res = Counter(res)
        # print("res")
        # print(res)
        resultant = resultant + res
        # print("resultant")
        # print(resultant)
    emotion_values = resultant. values()
    emotion_keys = []
    # print(emotion_values[0])
    # print(type(emotion_values))
    emotions = ["Angry", "Disgust", "Scared",
                "Happy", "Sad", "Surprised", "Neutral"]
    for x in resultant.keys():
        emotion_keys.append(emotions[x])
    print("emotion_keys")
    print(emotion_keys)
    print("after emotions")
    print(resultant)
    print("Testingggggggggggg")
    maz = 0
    u = -1
    overall_total = 0
    for item in emotion_values:
        u = u+1
        overall_total = overall_total + item
        if item > maz:
            maz = item
            zzz = u
    overall = maz
    overall_key = emotion_keys[zzz]
    zz = overall/overall_total * 100
    # text file
    garb = "python c java python ruby ruby ruby ruby rubyLorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of de Finibus Bonorum et Malorum (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, Lorem ipsum dolor sit amet.., comes from a line in section 1.10.32.There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text. All the Lorem Ipsum generators on the Internet tend to repeat predefined chunks as necessary, making this the first true generator on the Internet. It uses a dictionary of over 200 Latin words, combined with a handful of model sentence structures, to generate Lorem Ipsum which looks reasonable. The generated Lorem Ipsum is therefore always free from repetition, injected humour, or non-characteristic words etc"
    with open("./uploads/audio_text.txt") as f:
        bizz = str(f.read())
    print(bizz)
    print(len(re.findall(r'[a-zA-Z]+', bizz)))
    print("bizz iske uparrrrrrrrrrrrrrrrrrrrrrrrr")
    if(len(re.findall(r'[a-zA-Z]+', bizz)) <= 100):
        session["garbage"] = "True"
        with open("./uploads/audio_text.txt", "a") as text_file:
            print(str(garb), file=text_file)
    if 'garbage' in session:
        gar_val = session['garbage']
        print(gar_val)

    # cv, ans = textfile()

    result2 = loadProfile()
    dumpsProfile(result2)

    # node js
    ans = getSummary()

    chara={}

    for x in db.result.find({},{'_id':0,'personality.trait_id':1,'personality.name':1,'personality.percentile':1}):
        for y in x.values():
            print("yyyyy")
            print(y)
            p_percentile=[d['percentile']for d in y]
            p_name=[d['name']for d in y]

    print("percentile")
    print(p_percentile)

    print("name")
    print(p_name)


    

    # coherence
    if 'keywords' in session:
        k = session['keywords']
    co_cv, co_cv_d = coherence_cv(k)
    print("cv_coherence")
    co_ans, co_ans_d = coherence_ans(k)
    print("co_ans")
    cv_keys = co_cv_d.keys()
    ans_keys = co_ans_d.keys()
    print("WOWZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    # print("keys\n"+str(cv_keys)+"\n"+str(ans_keys))
    for i in ans_keys:
        print(i)
    print(type(cv_keys))
    print(type(ans_keys))
    ans_values = co_ans_d.values()
    cv_values = co_cv_d.values()
    print("dfasfsffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")
    for i in ans_values:
        print(i)
    # print("values\n"+str(cv_values)+"\n"+(ans_values))
    print(type(ans_values))
    print(type(cv_values))
    mayo = au.index(max(au))
    mayon = max(au)

    # session['cv_keys'] = cv_keys
    # session['gar_val'] = gar_val
    # session['ans_keys'] = ans_keys
    # session['ans_values'] = ans_values
    # session['cv_values'] = cv_values
    # session['content'] = content
    # session['cv'] = cv
    # session['ans'] = ans
    # session['co_cv'] = co_cv
    # session['co_ans'] = co_ans
    # session['contente'] = contente
    # session['emotion_keys'] = emotion_keys
    # session['emotion_values'] = emotion_values
    # session['au'] = au

    return render_template("report.html", garbage=gar_val, cv_keys=cv_keys, ans_keys=ans_keys, ans_values=ans_values,
                           cv_values=cv_values, prosody=content,  text_ans=ans, co_cv=co_cv,
                           co_ans=co_ans, emotions=contente, emotion_keys=emotion_keys, emotion_values=emotion_values, audio_values=au,
                           overall=overall, overall_key=overall_key, overall_total=overall_total, zz=zz, mayo=mayo, mayon=mayon,k=k[0],p_name=p_name,p_percentile=p_percentile)


@app.route('/loading')
def loading():
    # destroy()
    return render_template("loading.html")


# the below is for displaying next question---REnder it to the loading page in the else
@app.route('/check', methods=['POST'])
def check():
    key = check_audio()
    print("key check")
    return jsonify(key=key)


@app.route('/next')
def next():
    if session['questions']:
        myques = session['questions']
        popped = myques.pop(0)
        session['questions'] = myques
        print("idhar")
        print(popped)
        return render_template("student2.html", question=popped)
    else:
        return redirect(url_for("loading"))


@app.route('/finish')
def finish():
    mydir = app.config['UPLOAD_FOLDER']
    filelist = [f for f in os.listdir(mydir)]
    for f in filelist:
        print(f)
        os.remove(os.path.join(mydir, f))
    
    db.result.drop()
    
    return redirect(url_for("index"))


def finish_audio_files():
    mydir = app.config['UPLOAD_FOLDER']
    filelist = [f for f in os.listdir(mydir)]
    for f in filelist:
        if (f == "audio_coordinates.txt" or f == "audio_emotions.txt" or f == "audio_text.txt"):
            os.remove(os.path.join(mydir, f))


@app.route('/printpdf')
def printpdf():
    # if 'cv_keys' in session and 'gar_val' in session and 'ans_keys' in session and 'ans_values' in session and 'cv_values' in session:
    # and 'content' in session and 'cv' in session and 'ans' in session and 'co_cv' in session and 'co_ans' in session
    # and 'contente' in session and 'emotion_keys' in session and 'emotion_values' in session and 'au' in session:
    #     cv_keys = session['cv_keys']
    #     gar_val = session['gar_val']
    #     ans_keys = session['ans_keys']
    #     ans_values = session['ans_values']
    #     cv_values = session['cv_values']
    #     content = session['content']
    #     cv = session['cv']
    #     ans = session['ans']
    #     co_cv = session['co_cv']
    #     co_ans = session['co_ans']
    #     contente = session['contente']
    #     emotion_keys = session['emotion_keys']
    #     emotion_values = session['emotion_values']
    # au = session['au']
    # rendered = render_template('report.html')
    # # css="./static/css/style.css"
    # pdf=pdfkit.from_string(rendered,False)
    # print(pdf)
    # print(type(pdf))
    # response=make_response(pdf)
    # response.headers['Content-Type']='application/pdf'
    # response.headers['Content-Disposition']='inline; filename=output3.pdf'
    # return response
    # pdfkit.from_url('http://127.0.0.1:5000/afterloading', 'anol1.pdf')

    render_template_to_pdf('report.html', download=True, save=True)
    return redirect(url_for("index"))


if __name__ == "__main__":
    # app.run(debug=False)
    app.run(host='0.0.0.0', port=5000)

# host='0.0.0.0', port=5000
# host='http://127.0.0.1:5000/', debug=True, threaded=True
# this is reference Route -loads the questions in the session ----the below code has been added to route studentLogin
