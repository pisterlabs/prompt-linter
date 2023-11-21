#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Madhav (https://github.com/madhav-mknc)


from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.utils import secure_filename
from functools import wraps

from waitress import serve
from flask_cors import CORS

from google_auth_oauthlib.flow import Flow

from utils import *
import chatbot

import os
from dotenv import load_dotenv
load_dotenv()

# Initialzing flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# secret key
app.secret_key = os.getenv("FLASK_SECRET_KEY")  # Change this to a strong random key in a production environment
# app.secret_key = str(unique_id()).replace("-","")

# server address
HOST = "0.0.0.0"
PORT = 8080


# only logged in access
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# # GOOGLE API AUTHENTICATION
# state = session['state']
# flow = Flow.from_client_secrets_file(
#     'client_secret.json',
#     scopes=['https://www.googleapis.com/auth/drive.readonly'],
#     state=state)
# flow.redirect_uri = url_for('oauth2callback', _external=True)

# authorization_response = request.url
# flow.fetch_token(authorization_response=authorization_response)

# # Store the credentials in the session.
# # ACTION ITEM for developers:
# #     Store user's access and refresh tokens in your data store if
# #     incorporating this code into your real app.
# credentials = flow.credentials
# session['credentials'] = {
#     'token': credentials.token,
#     'refresh_token': credentials.refresh_token,
#     'token_uri': credentials.token_uri,
#     'client_id': credentials.client_id,
#     'client_secret': credentials.client_secret,
#     'scopes': credentials.scopes}
# flow = Flow.from_client_secrets_file('client_secret.json', SCOPES)
# flow.redirect_uri = REDIRECT_URI




# Routes below:
"""
/           => index
/login      => admin login page
/dashboard  => admin dashboard
/upload     => for uploading files
/handle_url => fetch data from URLs
/delete     => for deleting a uploaded file
/chatbot    => redirect to chatbot
/get_chat_response => for fetching response from the chatbot
/logout     => admin logout
"""


# index
@app.route('/')
def index():
    return render_template('index.html')


# login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('authenticated'):
        return redirect(url_for("dashboard"))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if is_authenticated(username, password):
            # Save the authenticated status in the session
            session['authenticated'] = True
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid credentials. Please try again.'
            return render_template('login.html', error=error)
    return render_template('login.html')


# dashboard
@app.route('/dashboard')
@login_required
def dashboard():
    # Check if the user is authenticated in the session
    if not session.get('authenticated'):
        return redirect(url_for('login'))

    files = list_stored_files()
    return render_template('dashboard.html', files=files)


# UPLOAD FILES from local storage
@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if request.method == 'POST':
        files = request.files.getlist('file')
        
        if not files:
            flash('No files selected')
            return redirect(url_for('dashboard'))

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(filename)

                status = upload_file_to_pinecone(filename)
                if os.path.exists(filename):
                    os.remove(filename)
                if status != "ok":
                    flash('Upload Limit Reached')
                    flash(status)
                    return redirect(url_for('dashboard'))
            else:
                flash('Invalid file type')
                return redirect(url_for('dashboard'))

        flash('Files uploaded successfully')
        return redirect(url_for('dashboard'))

# # GOOGLE DRIVE process files
# @app.route('/process_file_id', methods=['POST'])
# @login_required
# def process_file_id():
#     file_id = request.json.get('file_id')
#     # TODO: Your server side code to process files
#     print(file_id)
#     return jsonify({'message': 'File ID received'})

# # GOOGLE DRIVE route for OAuth 2.0 authorization
# @app.route('/login_google_drive')
# def login_google_drive():
#     auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline', redirect_uri=REDIRECT_URIS[0])
#     return redirect(auth_url)

# @app.route('/login_google_drive')
# def login_google_drive():
#     auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline')
#     return redirect(auth_url)

# # GOOGLE DRIVE Callback route for handling OAuth 2.0 response
# @app.route('/oauth2callback')
# def oauth2callback():
#     flow.fetch_token(authorization_response=request.url)
#     session['credentials'] = flow.credentials.to_json()
#     return redirect(url_for('dashboard'))


# # GOOGLE DRIVE upload files
# @app.route('/upload_google_drive', methods=['POST'])
# @login_required
# def upload_google_drive():
#     try:
#         if 'file' not in request.files:
#             flash('No file selected')
#             return redirect(url_for('dashboard'))

#         file = request.files['file']
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(url_for('dashboard'))

#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(filename)

#             drive_service = authenticate_google_drive(session)
#             file_id = upload_to_google_drive(drive_service, filename)
#             flash(f'File uploaded to Google Drive with ID: {file_id}')
#             return redirect(url_for('dashboard'))
#         else:
#             flash('Invalid file type')
#             return redirect(url_for('dashboard'))

#     except Exception as e:
#         flash(f'Error uploading to Google Drive: {str(e)}')
#         return redirect(url_for('dashboard'))



# UPLOAD FILES as txt scraped URL
@app.route('/handle_url', methods=['POST'])
@login_required
def handle_url():
    if request.method == 'POST':
        url = request.form.get('url')
        result_message = handle_urls(url)
        flash(result_message)
    return redirect(url_for('dashboard'))


# delete an uploaded file  
# @app.route('/delete/<filename>', methods=['POST'])
@app.route('/delete/<path:filename>')
@login_required
def delete(filename):
    delete_file_from_pinecone(filename)
    flash('File deleted successfully')
    return redirect(url_for('dashboard'))


# get response from chatbot
@app.route('/get_chat_response', methods=['POST'])
def get_chat_response():
    user_input = request.json['message']
    chat_history = request.json['conversationHistory']

    # truncate the chat_history
    chat_history.reverse()
    conversation = []
    for chat in chat_history:
        if len(str(conversation)) > 2000: # 2000 characters is the currect limit
            break 
        conversation.append(chat)
    chat_history = conversation[::-1]
    # for i in chat_history:
    #     print(i)

    response = chatbot.get_response(query=user_input, chat_history=chat_history)
    return jsonify({'message': response})

# chatbot
@app.route('/chatbot')
def chat():
    return render_template('chat.html')


# logout
@app.route('/logout')
def logout():
    # Clear the authenticated status from the session
    session.pop('authenticated', None)
    return redirect(url_for('index'))


# run server
def start_server():
    serve(app, host=HOST, port=PORT)

if __name__ == '__main__':
    start_server()
