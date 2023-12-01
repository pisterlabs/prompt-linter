from flask import Flask, request, jsonify, send_from_directory
import os
import openai
import json
import logging
import boto3
from dotenv import load_dotenv
from flask_cors import CORS
import sys

# Load environment variables
load_dotenv()

# Enter your dotenv file name with your API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Setup logging to console instead of a file
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Flask and CORS app setup
app = Flask(__name__, static_folder='dist', static_url_path='/')
CORS(app, resources={r"/ask": {"origins": "*"}})

# AWS S3 Setup (configured for Cyclic's S3 storage)
s3 = boto3.client('s3')

model_engine = "gpt-4"
model_prompt = "You are GPT-4, a large language model trained by OpenAI. Help answer questions and engage in conversation."
chat_history = []
max_history_tokens = 2000

# Save history function
def save_history(chat_history):
    s3.put_object(
        Body=json.dumps(chat_history),
        Bucket="cyclic-eager-battledress-yak-us-east-1",  
        Key="chat_history/my_chat_history.json"
    )
        
# Load history function
def load_history():
    try:
        my_file = s3.get_object(
            Bucket="cyclic-eager-battledress-yak-us-east-1",  
            Key="chat_history/my_chat_history.json"
        )
        return json.loads(my_file['Body'].read())
    except s3.exceptions.NoSuchKey:
        return []
    
# Response
def generate_response(prompt, model_engine, chat_history):
    if not prompt.strip():
        return ""

    conversation = "".join([f"{entry}\n" for entry in chat_history])

    try:
        response = openai.ChatCompletion.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": model_prompt},
                {"role": "user", "content": f"{conversation}User: {prompt}"}
            ],
            max_tokens=2000,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"]
        )
        text = response['choices'][0]['message']['content']
        text = text.strip()
        chat_history.append(f"Assistant: {text}")
    except openai.error.OpenAIError as e:
        text = f"Error: {e}"
        logging.error(text)
    
    return text
#Start React 
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


# Flask route to handle POST requests
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    response = generate_response(user_input, model_engine, chat_history)
    save_history(chat_history)  # Save the updated chat history to S3
    return jsonify({'response': response})


# Main block to run the Flask app
if __name__ == "__main__":
    chat_history = load_history()  # Load chat history from S3
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 3000)))

