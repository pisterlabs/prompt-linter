from flask import Flask, request, jsonify, render_template
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import os
import sqlite3
import ssl
import pickle
import json
from dotenv import load_dotenv

app = Flask(__name__)

DATA_FILE = 'data.pkl'
DB_FILE = 'conversations.db'
USER_DATA_FILE = 'user_data.json'
PAGE_SIZE = 1000 
current_pages = {}
cert_file = '/etc/nginx/ssl/server.crt.pem'
key_file = '/etc/nginx/ssl/MyChatGPT_WM.key'

def save_data(user_messages, user_data):
    data = {
        'user_messages': user_messages,
        'user_data': user_data,
    }
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(data, f)

def load_data():
    try:
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
            return data.get('user_messages', {}), data.get('user_data', {})
    except FileNotFoundError:
        return {}, {}

def setup_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY,
            user_id TEXT,
            convo_id INTEGER,
            step_id INTEGER,
            system_message TEXT,
            user_message TEXT,
            ai_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_user_data(user_data):
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(user_data, f)

def load_user_data():
    try:
        with open(USER_DATA_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

setup_db()

load_dotenv()  # loads environment variables from '.env' file

# Initialize the ChatOpenAI object with the API key
chat = ChatOpenAI(temperature=0)

# Initialize dictionaries to hold the messages for each user and the corresponding data
user_messages, user_data = load_data()
user_data_json = load_user_data()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', messages=user_messages.get('0', []))

@app.route('/chat', methods=['POST'])
def handle_chat():
    global user_messages, user_data
    # Get the message and user_id from the request
    user_input = request.json.get('message')
    user_id = request.json.get('user_id')

    # Initialize data for a new user
    if user_id not in user_messages:
        user_messages[user_id] = [SystemMessage(content="You are MyChatGPT, a helpful assistant dedicated to student's learning.")]
        user_data[user_id] = {
            0: {  # This is the initial conversation
                'system_message': user_messages[user_id][0],
                'steps': {}
            },
        }

    integer_keys = [k for k in user_data[user_id].keys() if isinstance(k, int)]
    current_convo = max(integer_keys) if integer_keys else 0
    current_step = max(user_data[user_id][current_convo]['steps'].keys()) if user_data[user_id][current_convo]['steps'] else 0

    if user_input == "clear":
        # Reset convo_id, step_id, and messages
        user_messages[user_id] = [SystemMessage(content="You are MyChatGPT, a helpful assistant dedicated to student's learning.")]
        user_data[user_id] = {
            'display_name': user_data[user_id].get('display_name', None),
            current_convo+1: {
                'system_message': user_messages[user_id][0],
                'steps': {}
            }
        }
        save_data(user_messages, user_data)
        return jsonify({"response": "Chat history cleared!"})

    # Create a new step with the user message
    user_messages[user_id].append(HumanMessage(content=user_input))
    user_data[user_id][current_convo]['steps'][current_step+1] = {
        'user_message': user_messages[user_id][-1],
        'ai_message': ""
    }

    # Generate a response using ChatGPT
    ai_response = chat(user_messages[user_id])
    user_messages[user_id].append(AIMessage(content=ai_response.content))

    # Update the AI message for the current step
    user_data[user_id][current_convo]['steps'][current_step+1]['ai_message'] = user_messages[user_id][-1]

    save_data(user_messages, user_data)  # Update pickle data file after every interaction

    # Save to SQLite database
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO conversations (
            user_id,
            convo_id,
            step_id,
            system_message,
            user_message,
            ai_response
        ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, current_convo, current_step+1, user_data[user_id][current_convo]['system_message'].content, user_input, ai_response.content))  
    conn.commit()
    conn.close()

    # Retrieve user's display name from the JSON database if available
    user_display_name = user_data_json.get(user_id, None)

    # Remove newline characters
    ai_response = ai_response.content.replace('\n', ' ')
    
    words = ai_response.split()
    current_pages[user_id] = []
    current_page = []
    for word in words:
        if sum(len(w) + 1 for w in current_page) + len(word) < PAGE_SIZE:  # +1 for each space
            current_page.append(word)
        else:
            current_pages[user_id].append(" ".join(current_page))
            current_page = [word]
    if current_page:  # if there are any words left in current_page
        current_pages[user_id].append(" ".join(current_page))
        
    # Send the reply back to the requester
    return jsonify({"response": f"{current_pages[user_id][0]}|{0}|{len(current_pages[user_id]) - 1}"})


    """# Send the reply back to the requester, displaying the user's display name if available
    if user_display_name:
        return jsonify({"response": f"{user_display_name}: {ai_response}"})
    else:
        return jsonify({"response": f"User: {ai_response}"})"""

@app.route('/check', methods=['GET'])
def check_user():
    # Get the user_id from the URL parameters
    user_id = request.args.get('user_id')

    # If the user_id does not exist, return an error message
    if user_id not in user_messages:
        return jsonify({"error": f"No data found for user_id: {user_id}"})

    # Get the user's messages
    messages = user_messages[user_id]

    # Retrieve user's display name from the JSON database if available
    user_display_name = user_data_json.get(user_id, None)

    # Format the chat history into an HTML representation
    formatted_history = ''
    for msg in messages:
        if isinstance(msg, HumanMessage):
            if user_display_name:
                formatted_history += f'<div class="message user">{user_display_name}: {msg.content}</div>'
            else:
                formatted_history += f'<div class="message user">User: {msg.content}</div>'
        elif isinstance(msg, AIMessage):
            formatted_history += f'<div class="message assistant">MyChatGPT: {msg.content}</div>'

    return render_template('chat_history.html', formatted_history=formatted_history)

@app.route('/register', methods=['POST'])
def register_user():
    global user_data_json

    # Get the user_id and display_name from the request
    user_id = request.json.get('user_id')
    display_name = request.json.get('display_name')

    # Reset the user's display name to "User" if display_name is empty
    if user_id and display_name == "/register":
        user_data_json[user_id] = "User"
        user_data[user_id]['display_name'] = "User"
    elif user_id and display_name:
        user_data_json[user_id] = display_name
        user_data[user_id]['display_name'] = display_name

    save_user_data(user_data_json)  # Save the updated user data to the JSON file

    return jsonify({"response": "User registered successfully!"})

@app.route('/chat/page/<int:page>', methods=['POST'])
def get_chat_page(page):
    # Get the user_id from the request
    user_id = request.json.get('user_id')

    # If the user_id or page number is not valid, return an error
    if user_id not in current_pages or page < 0 or page >= len(current_pages[user_id]):
        return jsonify({"error": "Invalid user_id or page number"})

    # Return the requested page
    return jsonify({"response": f"{current_pages[user_id][page]}|{page}|{len(current_pages[user_id]) - 1}"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    app.run(host='0.0.0.0', port=443, ssl_context=(cert_file, key_file))
