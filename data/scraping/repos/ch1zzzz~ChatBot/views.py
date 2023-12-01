# @author     : Jackson Zuo
# @time       : 10/5/2023
# @description: This module contains REST API of the app.

import threading
from queue import Queue
from flask import request, jsonify, render_template, Response
import os
import time
from langchain.callbacks import StreamingStdOutCallbackHandler
from werkzeug.utils import secure_filename
import logging
from app import app
from app.utils.decorators import require_valid_referer
from app.utils.embeddings import embedding
from app.utils.helper import allowed_file
from app.utils.question_answer_chain import getqa, get_session_id
from app.utils.streaming import StreamingStdOutCallbackHandlerYield, generate
from config import Config


@app.route('/')
def home():
    return render_template('index.html')


# page integrated with dialogflow
@app.route('/test')
def test():
    return render_template('test.html')


@app.route('/base')
def chatbot():
    return render_template('base.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/uploadFile', methods=['POST'])
def upload_file():
    """
    upload positions files

    Returns: success page

    """
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file and allowed_file(file.filename):  # check file suffix
        filename = secure_filename('nursejobs.csv')
        path = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(path)
        embedding()  # calculate the embeddings from new data
        return 'File successfully uploaded'

    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
@require_valid_referer
def predict():
    text = request.get_json().get("message")
    session_id = request.get_json().get("sessionId")
    logging.info(f"session_id: {session_id}, \n text: {text}")

    # create new chain
    if app.user_qa.get(session_id) is None:
        app.user_qa[session_id] = getqa()

    # set the conversation expire time to 10 minutes
    app.user_expiry[session_id] = time.time() + 600

    qa = app.user_qa[session_id]

    q = Queue()

    # streaming the output
    def streaming():
        callback_fn = StreamingStdOutCallbackHandlerYield(q)
        return qa.run({"question": text}, callbacks=[callback_fn, StreamingStdOutCallbackHandler()])

    threading.Thread(target=streaming).start()
    return Response(generate(q), mimetype='text/event-stream')


@app.route('/dialogflow/cx/receiveMessage', methods=['POST'])
def cx_receive_message():
    """
    Create a chain for each session and get answer from GPT

    Returns: Json to DialogFlow

    """

    data = request.get_json()
    session_id = get_session_id(data)
    if session_id is None:
        return jsonify(
            {
                'fulfillment_response': {
                    'messages': [
                        {
                            'text': {
                                'text': ['Something went wrong.'],
                                'redactedText': ['Something went wrong.']
                            },
                            'responseType': 'HANDLER_PROMPT',
                            'source': 'VIRTUAL_AGENT'
                        }
                    ]
                }
            }
        )

    if app.user_qa.get(session_id) is None:
        app.user_qa[session_id] = getqa()

    # set the conversation expire time to 10 minutes
    app.user_expiry[session_id] = time.time() + 600

    qa = app.user_qa[session_id]
    query_text = data['text']
    result = qa.run({"question": query_text})
    print(f"Chatbot: {result}")
    print("result finished!!!!!!!!")

    res = {"fulfillment_response": {"messages": [{"text": {"text": [result]}}]}}

    # Returns json
    return res
