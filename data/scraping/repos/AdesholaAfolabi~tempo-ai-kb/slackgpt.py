import os
import slack
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slackeventsapi import SlackEventAdapter
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, Response
import openai
from knowledge_base import doc_chat, load_embeddings
import threading
import requests

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define document files and directory for document embeddings
files = ["Ultimate_introduction_white_paper.pdf", "Planner-complete-guide-to-resource-planning.pdf", "roadmunk_support.pdf"]
persist_directory = "docs/chroma/"

# Create or load a vector database for document embeddings
vectordb = load_embeddings(persist_directory, files)

# Initialize the Slack client
client = slack.WebClient(token=os.getenv("SLACK_API_TOKEN"))
BOT_ID = client.api_call("auth.test")['user_id']

# Create a Flask app
app = Flask(__name__)

# Initialize SlackEventAdapter
slack_event_adapter = SlackEventAdapter(os.getenv("SIGNING_SECRET"), '/slack/events', app)

def some_processing(question, channel_id):
    """function for doing the actual work in a thread"""
    response = doc_chat(vectordb, model_name="gpt-3.5-turbo", question=str(question))  
    client.chat_postMessage(channel=channel_id, text=response)
    #return Response(), 200


@app.route('/ai-helper', methods=['POST'])
def ai_helper():
    """
    Respond to AI-related queries in Slack.
    """
    data = request.form
    #user_id = data.get('user_id')
    channel_id = data.get('channel_id')
    question = data.get('text')
    # starting a new thread for doing the actual processing    
    x = threading.Thread(
            target=some_processing,
            args=(question,channel_id,)
        )
    x.start()
    return "Processing information.... please wait"
if __name__ == '__main__':
    app.run(host="127.0.0.1", port=80, debug=True)
