# import necessary packages
import os
import cohere
from flask import Flask, render_template, request

# initialize cohere api client
co = cohere.Client(os.environ.get('COHEREAI_API_KEY'))

# initialize a conversation session id
cohere_chat_res_start = co.chat("Hi")
conv_session_id = cohere_chat_res_start.session_id

def getResponse(msg):
    cohere_chat_res = co.chat(msg, session_id=conv_session_id)
    return cohere_chat_res.reply

def chatbot_response(msg):
    res = getResponse(msg)
    return res

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
    app.run()