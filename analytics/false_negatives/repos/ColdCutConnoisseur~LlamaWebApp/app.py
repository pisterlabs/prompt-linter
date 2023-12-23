"""TBG Darwin OpenAI / Llamaindex website"""

import time
import datetime
import io
from threading import Thread

from flask import Flask, render_template, url_for, session, redirect, flash, request, g, Response, send_file
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField
from wtforms.validators import DataRequired
from werkzeug.security import check_password_hash
from flask_bootstrap import Bootstrap
#import requests
#from lxml import etree

from flask_mobility import Mobility
from flask_mobility.decorators import mobile_template

import config_file as cf
from index_helpers import (load_index,
                           combine_pangolin_indices,
                           combine_darwin_indices,
                           load_first_darwin_index_chunk,
                           load_second_darwin_index_chunk,
                           load_third_darwin_index_chunk,
                           load_fourth_darwin_index_chunk,
                           load_fifth_darwin_index_chunk,
                           load_sixth_darwin_index_chunk,
                           load_seventh_darwin_index_chunk,
                           load_eighth_darwin_index_chunk)


app = Flask(__name__)
app.config['SECRET_KEY'] = cf.WTF_KEY
app.config['DARWIN_DICT'] = {}
app.config['DARWIN_INDEX'] = None

bootstrap = Bootstrap(app)
Mobility(app)

P_HASH_CHECK = cf.PW_HASH

class QueryForm(FlaskForm):
    query_string = StringField('Enter your question here:', validators=[DataRequired()])
    submit = SubmitField('Query')

class LoginForm(FlaskForm):
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Log In')


def check_login_state(session_obj):
    """Helper"""
    try:
        current_state = session_obj['UserSignedIn']

    except KeyError:
        session_obj['UserSignedIn'] = False
        current_state = session_obj['UserSignedIn']

    return current_state


def load_pangolin_resources():
    storage_folder1 = "/pangolin_index_storage_folder1"
    storage_folder2 = "/pangolin_index_storage_folder2"
    storage_folder3 = "/pangolin_index_storage_folder3"
    storage_folder4 = "/pangolin_index_storage_folder4"
    storage_folder5 = "/pangolin_index_storage_folder5"
    storage_folder6 = "/pangolin_index_storage_folder6"
    storage_folder7 = "/pangolin_index_storage_folder7"
    storage_folder8 = "/pangolin_index_storage_folder8"
    storage_folder9 = "/pangolin_index_storage_folder9"
    storage_folder10 = "/pangolin_index_storage_folder10"
    storage_folder11 = "/pangolin_index_storage_folder11"
    storage_folder12 = "/pangolin_index_storage_folder12"
    storage_folder13 = "/pangolin_index_storage_folder13"
    storage_folder14 = "/pangolin_index_storage_folder14"
    storage_folder15 = "/pangolin_index_storage_folder15"
    storage_folder16 = "/pangolin_index_storage_folder16"
    storage_folder17 = "/pangolin_index_storage_folder17"
    storage_folder18 = "/pangolin_index_storage_folder18"
    storage_folder19 = "/pangolin_index_storage_folder19"
    storage_folder20 = "/pangolin_index_storage_folder20"
    storage_folder21 = "/pangolin_index_storage_folder21"
    storage_folder22 = "/pangolin_index_storage_folder22"
    storage_folder23 = "/pangolin_index_storage_folder23"
    storage_folder24 = "/pangolin_index_storage_folder24"
    storage_folder25 = "/pangolin_index_storage_folder25"
    storage_folder26 = "/pangolin_index_storage_folder26"

    loaded_index = combine_pangolin_indices(
                                storage_folder1,
                                storage_folder2,
                                storage_folder3,
                                storage_folder4,
                                storage_folder5,
                                storage_folder6,
                                storage_folder7,
                                storage_folder8,
                                storage_folder9,
                                storage_folder10,
                                storage_folder11,
                                storage_folder12,
                                storage_folder13,
                                storage_folder14,
                                storage_folder15,
                                storage_folder16,
                                storage_folder17,
                                storage_folder18,
                                storage_folder19,
                                storage_folder20,
                                storage_folder21,
                                storage_folder22,
                                storage_folder23,
                                storage_folder24,
                                storage_folder25,
                                storage_folder26
                                )
    return loaded_index


@app.route('/login', methods=['GET', 'POST'])
@mobile_template('{mobile/}login.html')
def check_login_page(template):

    user_logged_in = check_login_state(session)

    if user_logged_in:
        return redirect('/')
    
    else:
        login_form = LoginForm()

        if login_form.validate_on_submit():
            pword = login_form.password.data

            session['UserSignedIn'] = check_password_hash(P_HASH_CHECK, pword)

            if session['UserSignedIn']:
                return redirect('/')
            
            else:
                flash('Invalid password!')

        return render_template(template, form=login_form)

@app.route('/')
@mobile_template('{mobile/}darwin_loading.html')
def darwin_query(template):
    user_logged_in = check_login_state(session)
    
    if user_logged_in:
        if app.config['DARWIN_INDEX']:
            return redirect('/darwin')

        else:
            return render_template(template)
    
    else:
        return redirect('/login')
    
@app.route('/loading_screen2')
@mobile_template('{mobile/}darwin_loading2.html')
def load_chunk2(template):
    return render_template(template)

@app.route('/loading_screen3')
@mobile_template('{mobile/}darwin_loading3.html')
def load_chunk3(template):
    return render_template(template)

@app.route('/loading_screen4')
@mobile_template('{mobile/}darwin_loading4.html')
def load_chunk4(template):
    return render_template(template)

@app.route('/loading_screen5')
@mobile_template('{mobile/}darwin_loading5.html')
def load_chunk5(template):
    return render_template(template)

@app.route('/loading_screen6')
@mobile_template('{mobile/}darwin_loading6.html')
def load_chunk6(template):
    return render_template(template)

@app.route('/loading_screen7')
@mobile_template('{mobile/}darwin_loading7.html')
def load_chunk7(template):
    return render_template(template)

@app.route('/loading_screen8')
@mobile_template('{mobile/}darwin_loading8.html')
def load_chunk8(template):
    return render_template(template)
    
@app.route('/load_darwin_index')
def old_darwin_load():
    return "Finished!"

@app.route('/load_darwin_resources_step1')
def get_first_dict():
    print("Step1")
    first_dict = load_first_darwin_index_chunk()
    app.config['DARWIN_DICT'] = app.config['DARWIN_DICT'] | first_dict
    return "Finished!"

@app.route('/load_darwin_resources_step2')
def get_second_dict():
    print("Step2")
    second_dict = load_second_darwin_index_chunk()
    app.config['DARWIN_DICT'] = app.config['DARWIN_DICT'] | second_dict
    return "Finished!"

@app.route('/load_darwin_resources_step3')
def get_third_dict():
    print("Step3")
    new_dict = load_third_darwin_index_chunk()
    app.config['DARWIN_DICT'] = app.config['DARWIN_DICT'] | new_dict
    return "Finished!"

@app.route('/load_darwin_resources_step4')
def get_fourth_dict():
    print("Step4")
    new_dict = load_fourth_darwin_index_chunk()
    app.config['DARWIN_DICT'] = app.config['DARWIN_DICT'] | new_dict
    return "Finished!"

@app.route('/load_darwin_resources_step5')
def get_fifth_dict():
    print("Step5")
    new_dict = load_fifth_darwin_index_chunk()
    app.config['DARWIN_DICT'] = app.config['DARWIN_DICT'] | new_dict
    return "Finished!"

@app.route('/load_darwin_resources_step6')
def get_sixth_dict():
    print("Step6")
    new_dict = load_sixth_darwin_index_chunk()
    app.config['DARWIN_DICT'] = app.config['DARWIN_DICT'] | new_dict
    return "Finished!"

@app.route('/load_darwin_resources_step7')
def get_seventh_dict():
    print("Step7")
    new_dict = load_seventh_darwin_index_chunk()
    app.config['DARWIN_DICT'] = app.config['DARWIN_DICT'] | new_dict
    return "Finished!"

@app.route('/load_darwin_resources_step8')
def get_eighth_dict():
    print("Step8")
    new_dict = load_eighth_darwin_index_chunk()
    app.config['DARWIN_DICT'] = app.config['DARWIN_DICT'] | new_dict
    return "Finished!"

@app.route('/create_and_set_darwin_index')
def set_index():
    graph = combine_darwin_indices(app.config['DARWIN_DICT'])
    app.config['DARWIN_INDEX'] = graph
    return redirect('/darwin')

@app.route('/darwin', methods=['GET', 'POST'])
@mobile_template('{mobile/}index.html')
def index(template):

    user_logged_in = check_login_state(session)
    
    if user_logged_in:
        loaded_index = app.config['DARWIN_INDEX']

        if loaded_index is None:
            return redirect('/')

        query_text = None
        query_response = ''

        q_form = QueryForm()

        if q_form.validate_on_submit():
            query_text = q_form.query_string.data

            if loaded_index: # Filter out bad index loads (which will return 'None')
                query_prompt = f"The 'user query' is '{query_text}'."+\
                                " Please only respond to the 'user query' with related information" +\
                                " found only in the provided query_engine object.  If related information" +\
                                " is not found within those 'chunks' then respond with" +\
                                " 'not found'."
                query_engine = loaded_index.as_query_engine()
                query_response = query_engine.query(query_prompt)

        return render_template(template, form=q_form, query_response=query_response)
    
    else:
        return redirect('/login')

@app.route('/pangolin')
@mobile_template('{mobile/}loading.html')
def pangolin_query(template):
    user_logged_in = check_login_state(session)
    
    if user_logged_in:
        return render_template(template)
    
    else:
        return redirect('/login')

@app.route('/load_pangolin_index')
def old_load():
    return "Finished!"

@app.route('/pangolin_loaded', methods=['GET', 'POST'])
@mobile_template('{mobile/}pangolin.html')
def load_pangolin_index(template):
    user_logged_in = check_login_state(session)
    
    if user_logged_in:
    
        p_index = load_pangolin_resources()
        query_text = None
        query_response = ''

        q_form = QueryForm()

        if q_form.validate_on_submit():
            query_text = q_form.query_string.data

            if p_index: # Filter out bad index loads (which will return 'None')
                print("Index exists!")
                query_engine = p_index.as_query_engine()
                
                query_prompt = f"The 'user query' is '{query_text}'."+\
                                " Please only respond to the 'user query' with information" +\
                                " found only in the provided 'chunks', 'chunk1' to 'chunk26'.  If the information" +\
                                " is not found within those 'chunks' then respond with" +\
                                " 'not found'."
                
                query_response = query_engine.query(query_prompt)

        return render_template(template, form=q_form, query_response=query_response)
    
    else:
        return redirect('/login')

@app.route('/darwin_site_preview.png')
def prep_mobile_image():
    preview_image_url = "./static/images/social_preview_square.png"
    return send_file(preview_image_url, mimetype='image/png')