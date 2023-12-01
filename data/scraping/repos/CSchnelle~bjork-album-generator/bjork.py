import openai
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)
openai.api_key = 'enter your chat gpt api key here!'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_bjork', methods=['POST'])
def generate_bjork():
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="You are a helpful assistant that provides an album name that sounds like it could be a Bjork album name. Give me an album name that sounds like a Bjork album name.",
        max_tokens=50,
    )
    album_name = response.choices[0].text.strip()
    return jsonify({'bjork':album_name})

if __name__ == '__main__':
    app.run()