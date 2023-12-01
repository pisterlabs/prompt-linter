import os
from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for, jsonify)
from flask_httpauth import HTTPBasicAuth
from chatbot_factory import ChatBotFactory
from langchain import (OpenAI,HuggingFaceHub, Cohere)
from chatbot_settings import ChatBotSettings
import os
from langchain.chains.conversation.memory import (ConversationBufferMemory,
                                                  ConversationSummaryMemory,
                                                  ConversationBufferWindowMemory,
                                                  ConversationKGMemory)
from langchain.llms import Cohere



app = Flask(__name__)
auth = HTTPBasicAuth()

# Define a dictionary to store valid usernames and passwords

@auth.verify_password
def verify_password(username, password):
    users = ChatBotSettings().get_website_users()
    if username in users and users[username] == password:
        return username

@app.route('/chatbot')
@auth.login_required
def chatbot():
    print('Request for chatbot page received')
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    # Get the JSON payload from the request
    data = request.get_json()

    # Extract the message from the JSON payload
    message = data.get('message')

    chatbotType = data.get('chatBotType')
    
    #Initialize services
    chatbot_factory = ChatBotFactory()

    chatbot = chatbot_factory.create_service("BotConversationChain", ChatBotSettings(llm=ChatBotFactory().llms["ChatOpenAI"],memory=ConversationBufferMemory(), tools=['serpapi','wolfram-alpha']))

    response = chatbot.get_bot_response(message)
    # Return the response as a JSON object
    return jsonify({'response': response})

@app.route('/mobile')
@auth.login_required
def mobile():
    print('Request for mobile_chatbot page received')
    return render_template('mobile_chatbot.html')

@app.route('/')
@auth.login_required
def index():
    print('Request for index page received')
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')



if __name__ == '__main__':
    app.run()