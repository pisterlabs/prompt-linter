import openai

import os
from flask import Flask, render_template, request, redirect, url_for, flash, session
import json 
from flask_cors import CORS

#json credentials


with open('credentials.json') as f:
    data = json.load(f)

openai.api_key = data['TOKEN']









# Make a request to the OpenAI Completion API, specifying the `gpt-4` engine

# Get the generated text
def msgresponse(message):
    msg = f" explain the brute force and optimized solution to {message} in pseudocode"
    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": message
    }],
    )
    msg = response["choices"][0]["message"]["content"]
    return {"msg":msg}


def coderesponse(message,lang):
    
    msg = f" explain the brute force and optimized solution to {message} in {lang} programming language"
    
    response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": message
    }],
    )
    msg = response["choices"][0]["message"]["content"]
    return {"msg":msg}


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "Hello World"

#return it as a json object
@app.route('/chat/<message>', methods=['GET'])
def chat(message):
    return msgresponse(message)
'''

127.0.0.1:5000/chat/Hello

'''


@app.route('/code/<message>/<lang>', methods=['GET'])
def code(message,lang):
    return coderesponse(message,lang)









if __name__ == '__main__':
    app.run(debug=True)







