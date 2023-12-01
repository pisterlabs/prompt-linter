from flask import Flask,request
import os
import openai

app = Flask(__name__)

openai.api_key = os.environ.get('OPENAI_KEY')


@app.route('/')
def index():
    return "<h1>Hello, World!</h1>"

@app.route('/chatgpt')
def chatgpt():
    args = request.args
    message =args.get("message")
    print(message)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}]
    )
    return completion['choices'][0]['message']['content']


