from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import openai
import datetime

#authentication
openai.organization = My organization number
openai.api_key = My API Key

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # add CORS support

# Define endpoint for ChatGPT
@app.route('/qa', methods=['POST'])
def chatgpt():
    # Get the user's question from the POST request
    question = request.form['question']

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\nQ: Who is [artist_name]?\nA: [artist_bio]\n\nQ: What is [artist_name]'s most famous artwork?\nA: [artwork_info]\n\nQ: What style of art is [artist_name] associated with?\nA: [artistic_style_info]\n\nQ: When was [artist_name] born?\nA: [birth_info]\n\nQ: When did [artist_name] die?\nA: [death_info]\n\nQ: "+question+"\nA:",
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )

    # Return the response text to the user
    return response.choices[0].text

# Use the 'application' object to define the entry point for your Flask app
application = app
