import re
from threading import Thread
import openai
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from flask import Flask, request, make_response, jsonify, g
import sqlite3
from cryptography.fernet import Fernet
import base64
import traceback, sys
import config
import tiktoken
import gc

# lets make gil more streamlined
gc.collect(2)
gc.freeze()
allocs, g1, g2 = gc.get_threshold()
gc.set_threshold(100_000, g1*5, g2*10)

# Initialize Flask app
app = Flask(__name__)

global client_message_id_list
client_message_id_list = []

# dbapi.py - part here ---
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect("chat_history.db")
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()

@app.teardown_appcontext
def teardown_db(e=None):
    close_db(e)

# Database setup
def setup_database():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content BLOB NOT NULL,
            client_msg_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_id ON messages(user_id);
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversation_id ON messages(conversation_id);
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_content ON messages(content);
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_client_msg_id ON messages(client_msg_id);
    """)

    conn.commit()
    conn.close()

class DBClient:
    def __init__(self):
        self.encryption_key = config.ENCRYPTION_KEY
        setup_database()

    def encrypt(self, text):
        f = Fernet(self.encryption_key)
        encrypted_text = f.encrypt(text.encode())
        return base64.b64encode(encrypted_text)

    def decrypt(self, cipher_text):
        f = Fernet(self.encryption_key)
        decoded_cipher_text = base64.b64decode(cipher_text)
        return f.decrypt(decoded_cipher_text).decode()

    def save_message(self, user_id, conversation_id, role, content, client_msg_id):
        if config.ENCRYPTION_SET:
            encrypted_content = self.encrypt(content)
        else:
            encrypted_content = content
        
        db = get_db()
        db.execute("""
            INSERT INTO messages (user_id, conversation_id, role, content, client_msg_id)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, conversation_id, role, encrypted_content, client_msg_id))

        db.commit()

    def get_conversation_history(self, user_id, conversation_id):
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            SELECT role, content FROM messages
            WHERE user_id = ? AND conversation_id = ?
            ORDER BY timestamp DESC LIMIT 15
        """, (user_id, conversation_id,))

        history = []
        for row in cursor.fetchall():
            role, content = row
            if config.ENCRYPTION_SET:
                decrypted_content = self.decrypt(content)
            else:
                decrypted_content = content
            history.append({"role": role, "content": decrypted_content})

        return history
    
    def get_message(self, client_msg_id):
        db = get_db()
        cursor = db.cursor()
        cursor.execute("""
            SELECT * FROM messages
            WHERE client_msg_id = ?
        """, (client_msg_id,))

        if cursor.fetchone():
            return True
        else:
            return False

# oaicon.py - part here ---
class SecureOpenAIChatGPTClient:
    def __init__(self, db_client):
        self.db_client = db_client

    def count_tokens(self, message=None, encoding_model=None):
        if not encoding_model:
            encoding_model = config.OPENAI_ENGINE # as default
        encoding = tiktoken.encoding_for_model(encoding_model)
        num_tokens = len(encoding.encode(message))
        return num_tokens

    def send_message(self, message, user_id, conversation_id, client_msg_id):
        try:
            # remove from the list if more than 150 messages
            if len(client_message_id_list) > 150:
                del client_message_id_list[50:]
            
            # check memory first if message already exists
            if client_msg_id in client_message_id_list:
                return False
            
            token_count = self.count_tokens(message)

            if token_count > 4096:
                return f"Message is too long {token_count}/4096 tokens to reply back to you. Please try again with a shorter message."

            # check db if message already exists
            if self.db_client.get_message(client_msg_id):
                return False
            
            client_message_id_list.append(client_msg_id)
            conversation_history = self.db_client.get_conversation_history(user_id, conversation_id)
            
            context_total = ''
            session_messages = []
            for i in range(len(conversation_history)):
                role = "assistant" if i % 2 else "user"
                if role == "assistant":
                    context_total += conversation_history[i]['content']
                    if self.count_tokens(context_total+message) > 4096:
                        break
                    
                    session_messages.append({"role": role, "content": conversation_history[i]['content']})
   
            context_total = '' # some clearing
            
            # append the latest to last in list
            session_messages.append({"role": "user", "content": message})
            try:
                response = openai.ChatCompletion.create(
                    model=config.OPENAI_ENGINE,
                    messages=session_messages,
                    temperature=0.7,
                    max_tokens=4096,
                    stream=True,
                )
            
            except openai.error.InvalidRequestError as __request_error:
                return str(__request_error)
            
            except openai.error.APIError as __api_error:
                return str(__api_error)

            collected_messages = []
            
            for chunk in response:
                chunk_message = chunk['choices'][0]['delta']  # extract the message
                collected_messages.append(chunk_message)  # save the message
            
            _message = ''.join([m.get('content', '') for m in collected_messages])
            
            self.db_client.save_message(user_id, conversation_id, "user", str(message), client_msg_id)
            self.db_client.save_message(user_id, conversation_id, "assistan", str(_message), client_msg_id)

            return _message

        except Exception as e:
            print(traceback.print_exc(file=sys.stdout))
            return "ERROR"

# flask_bot.py - part here ---
def format_response(response):
    """
    Format the OpenAI Chatbot's response in Slack's code block style
    if the response contains code.
    """
    return response

@app.route('/', methods=['POST'])
def handle_root():
    event = request.json
    try:
        #print(event)
        token = event.get("token")
        challenge = event.get("challenge")
        event_type = event.get("type")

        if event_type == "url_verification":
            if token == config.VERIFICATION_TOKEN:
                resp = make_response(challenge, 200)
                resp.headers['Content-Type'] = 'text/plain'
                return resp  

        else:
            return make_response("", 404)
    
    except Exception as e:
        print(e)
        return make_response("", 404)

@app.route('/slack/events', methods=['POST'])
def handle_event():
    event = request.json
    event_type = event.get("type")
    #print(event)

    # slack verification event handling
    if event_type == "url_verification":
        token = event.get("token")
        challenge = event.get("challenge")
        
        if token == config.VERIFICATION_TOKEN:
            resp = make_response(challenge, 200)
            resp.headers['Content-Type'] = 'text/plain'
            return resp
        else:
            return make_response("", 404)    
    
    # slack event handling
    _event = event.get("event")
    if _event:
        _event_type = _event["type"]

    if _event_type == "app_mention":
        handle_mention(event)
    elif _event_type == "message":
        handle_message(_event)
    else:
        return make_response("", 404)

    return make_response("", 200)

def handle_mention(event):
    event = event["event"]
    #prompt = re.sub('\\s<@[^, ]*|^<@[^, ]*', '', event['text'])
    try:
        #response = chatbot.ask(prompt)
        response = f"I rather would like to talk privately with you, lets have a 1:1 instead of this public channel. ok?"
        send = format_response(response)
    except Exception as e:
        print(e)
        send = "We're experiencing exceptionally high demand. Please, try again."

    say(event, send)

def handle_message(event):
    if "bot_id" in event:
        return
    # make sure we have a the global oaic_client in use
    global oaic_client
    
    # get the user_id, channel_id, text, thread_ts and ts
    user_id = event["user"]
    channel_id = event["channel"]
    text = event["text"]
    thread_ts = event.get("thread_ts", None)
    ts = event["ts"]
    client_msg_id = event["client_msg_id"]

    # always reply in users thread
    if thread_ts:
        conversation_id = thread_ts
    else:
        conversation_id = ts

    # remove the @mention from the text
    _text = re.sub('\\s<@[^, ]*|^<@[^, ]*', '', text)
    try:
        for i in range(3):
            response = oaic_client.send_message(_text, user_id, conversation_id, client_msg_id)
            if response == False:
                return
            if response != "ERROR":
                send = format_response(response)
                break
        if response == "ERROR":
            response = "problems with openai API, please try again what you wrote"
            send = format_response(response)

    except Exception as e:
        print(e)
        send = "We're experiencing exceptionally high demand. Please, try again."

    if thread_ts:
        response = client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=send
        )
    else:
        #response = client.conversations_open(users=[user_id])
        #channel_id = response["channel"]["id"]
        response = client.chat_postMessage(
            channel=channel_id,
            thread_ts=ts,
            text=send
        )

    #say(response["message"]["text"])

# this is not needed anymore - should be removed
def say(event, text):
    channel_id = event["channel"]
    thread_ts = event.get("thread_ts", None)
    ts = event["ts"]

    if thread_ts:
        response = client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=text
        )
    else:
        response = client.chat_postMessage(
            channel=channel_id,
            thread_ts=ts,
            text=text
        )

def load_key_from_file(filename):
    with open(filename, "rb") as key_file:
        key = key_file.read()
    return key

# Initialize DB client
db_client = DBClient()

# Initialize Slack's Web API client
client = WebClient(token=config.SLACK_BOT_TOKEN)

# Initialize OpenAI's Chatbot client
openai.api_key = config.OPENAI_API_KEY
global oaic_client
oaic_client = SecureOpenAIChatGPTClient(db_client)

# only with DNS on, comment out for ngrok
full_chain_path = "/fullchain.pem"
priv_key_path = "/privkey.pem"


if __name__ == "__main__":
    #app.run(port=4000) # with ngrok
    app.run(host='0.0.0.0', port=443, ssl_context=(full_chain_path, priv_key_path))