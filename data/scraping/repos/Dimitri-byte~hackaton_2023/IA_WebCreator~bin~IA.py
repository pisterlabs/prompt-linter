import json
import os
import openai
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

with open("/srv/IA_WebCreator/bin/api_key.txt", "r") as file:
    open_api_key = file.read().strip()
contextPrompt = 'As a professional front end developer, create an HTML and CSS using link rel skeleton with responsive design using Bootstrap and fontawesome for the following scenario.'
resumeTitlePrompt = 'Also, please create a summary of the following scenario to be returned as the title'
formattingPrompt = 'Return the answer as a JSON object with the following format: '
jsonFormatString = '[{"htmlCode": "html code", "title": "summary of the scenario"}]'
cleaningUpJsonPrompt = 'Remove all occurrences of \\n and \\ and \\\ in your response.'

def generate_text(prompt):
    api_key = open_api_key
    model = "gpt-3.5-turbo"
    endpoint = "https://api.openai.com/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": contextPrompt + resumeTitlePrompt},
            {"role": "user", "content": prompt},
            {"role": "user", "content": formattingPrompt + jsonFormatString}
        ],
        "max_tokens": 1000,
        "n": 1,
        "stop": None,
        "temperature": 0.5,
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(endpoint, headers=headers, json=payload)
    if response.status_code == 200:
        completions = response.json()["choices"][0]["message"]["content"]
        #print(f"completions: {completions}")   
        return completions
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(f"response.text: {response.text}")
        return None

app = Flask(__name__)
CORS(app)

@app.route('/generate-text', methods=['POST'])
def generate_text_api():
    prompt = request.json.get('prompt')
    generated_text = generate_text(prompt)
    generated_text = generated_text.replace('\\n', '\n')
    generated_text = generated_text.replace('\\t', '\t')
    generated_text = generated_text.replace('\\\"', '\"')
    print(f"generated_text: {generated_text}")  

    response = {
        'generated_text': generated_text
    }

    #return jsonify(response)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0')