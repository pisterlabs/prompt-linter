import os
import time
from flask import Flask, render_template, redirect, url_for, make_response
from forms import AddForm, DelForm, LoginForm
from newapp import Database
from cohere_mood_training import get_emotion
from flask import request

time.clock = time.time
app = Flask (__name__)
# secret key for forms
app.config['SECRET_KEY'] = 'mysecretkey'

############################ Forms ##############################
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    loginForm = LoginForm()
    
    if loginForm.validate_on_submit():
        email = loginForm.email.data
        password = loginForm.password.data
        user = Database().login_user(password, email)
        # put user id into cookie
        if(user == None):
            return render_template('login.html', form=loginForm)
        resp = make_response(redirect(url_for('add_entry')))
        resp.set_cookie('userid', str(user.userid))
        return resp
    
    return render_template('login.html', form=loginForm)

@app.route('/add', methods=['GET', 'POST'])
def add_entry():
    form = AddForm()

    if form.validate_on_submit():
        # get userid from cookie
        userid = request.cookies.get('userid')
        jname = form.jname.data 
        content = form.content.data
        emotion = get_emotion(content)  # Assuming get_emotion returns the emotion based on content
        print('AI: %s' %emotion)
        eid = -1
        emotions = Database().get_all_emotions()
        for em in emotions:
            if em.emotion == emotion:
                eid = em.emotionid
                break
        if eid != -1:
            jid = Database().insert_journal(userid, eid, jname, content)
            print('jid: %s' %str(jid))
        # Perform the necessary actions with the data, for example, store it in a database

        # return redirect(url_for('home.html'))  # Redirect to the index page after successful form submission
    else:
        print(form.errors)
    return render_template('write_copy.html',form=form)

#################### TODO: Dynamically assign Jname ##############
@app.route('/list')
def list_entries():
    # get userid from cookie
    userid = request.cookies.get('userid')
    journal_ids = Database().get_all_journal_ids(userid)
    journal_data = []
    for jid in journal_ids:
        journal_data.append(Database().get_journal_by_id(jid))
    
    for j in journal_data:
        eid = j.eid
        j.music = Database().get_music(eid)
        j.exercises = Database().get_exercises(eid)
        j.advices = Database().get_advices(eid)
    
    return render_template('recs_copy.html', all_entries = journal_data)

@app.route('/delete', methods=['GET', 'POST'])
def del_entry():
    pass

if __name__ == '__main__':
    app.run(debug=True)