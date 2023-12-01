from flask import Flask, render_template, request, jsonify
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage  # Make sure to import HumanMessage
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare_responses():
    try:
        model_names = request.json['model_names']
        openai_api_key = request.json['openai_api_key']
        question = request.json['question']
        answers = []

        for model_name in model_names:
            chat = ChatOpenAI(model=model_name, openai_api_key=openai_api_key)
            messages = [HumanMessage(content=question)]  # Use HumanMessage here
            response = chat(messages)
            answers.append(response.content)

        return jsonify(answers=answers)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
