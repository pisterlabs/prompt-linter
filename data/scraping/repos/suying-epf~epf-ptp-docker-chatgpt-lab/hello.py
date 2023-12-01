from flask import Flask, request
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
    message = args.get("message")
    print(message)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": message}]
    )
    return completion['choices'][0]['message']['content']


@app.route('/code')
def generate_code():
    language = request.args.get('language')
    content = request.args.get('content')
    message = f"Generate code in {language} that {content}"

    # TODO: Use ChatGPT to generate code in the specified language.
    completion = openai.Completion.create(
        engine="text-davinci-002",
        prompt=message,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5
    )
    result = completion.choices[0].text.strip()

    return result
