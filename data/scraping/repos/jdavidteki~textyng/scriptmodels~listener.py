import openai
import os
from flask import Flask, request
from flask_cors import CORS
from datetime import datetime

openai.api_key = os.environ["OPENAI_API_KEY"]

app = Flask(__name__)
CORS(app)

conversation_history = []

def ask(question, conversation_history):
    openai_response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=(f"{conversation_history}User: {question}\nAI:"),
        temperature=0.5,
        max_tokens=150,
        n=1,
        stop=["\nUser:", "AI:"]
    )

    message = openai_response.choices[0].text.strip()
    conversation_history += f"\nUser: {question}\nAI: {message}\n{str(datetime.now())}\n"
    with open("conversation_history.txt", "w") as f:
        f.write(conversation_history)

    return message

@app.route('/ask', methods=['POST'])
def handle_ask():
    data = request.get_json()
    input_text = data['inputText']
    conversation_history_file = open("conversation_history.txt", "r")
    conversation_history = conversation_history_file.read()
    conversation_history_file.close()
    return ask(input_text, conversation_history)

if __name__ == '__main__':
    app.run(debug=True)
