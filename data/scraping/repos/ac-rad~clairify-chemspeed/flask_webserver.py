from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import os
from threading import Thread, Lock
import json
import openai
from generate_chatgpt_func_call import convert

def translate(input):
    """Function that translates the input to Python"""
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    #openai.organization = os.environ.get("OPENAI_ORGANIZATION_ID")
    
    # Get the output
    convert(input, socketio)


# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(32)
app.config["SESSION_COOKIE_SECURE"] = True
socketio = SocketIO(app)

# Global variables
thread = None
thread_lock = Lock()
input = ""


def run_translation(input):
    """Function that runs the translation in a separate thread."""
    global thread
    with thread_lock:
        thread = Thread(target=translate, args=(input,))
        thread.start()

@app.route("/", methods=["GET", "POST"])
def index():
    """Function that renders the index page."""
    global input
    if request.method == "POST":              
        input = request.form["input_field"]
        run_translation(input)

    return render_template("index.html", input=input)

socketio.run(app, debug=True, port=3000)
