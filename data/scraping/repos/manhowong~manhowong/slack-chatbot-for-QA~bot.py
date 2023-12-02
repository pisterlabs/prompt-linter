import os
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request
from agent import genius
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma

# Load environment variables from .env file
load_dotenv(find_dotenv('private/.env'))

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the Flask app
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

# Set up your base vector base
# Get Cohere embedding model (you may also use other models)
embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
text = "INSERT YOUR TEXT HERE"
db = Chroma.from_texts(text, embeddings)

@app.event("app_mention")
def handle_mentions(body, say):
    """
    Event listener for mentions in Slack.
    This function send a response to the event when the bot is mentioned.
    body (dict): event data
    say (callable): function for sending a response
    """

    # Get query from event (i.e. user's message)
    query = body["event"]["text"]
    mention = f"<@{SLACK_BOT_USER_ID}>"
    query = query.replace(mention, "").strip()
    
    # Response
    say('One moment please...')
    ans = genius(query, db)  # genius is the engine of the bot!
    say(ans)

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler.
    """
    return handler.handle(request)  # result of handling the request

# Run the Flask app
if __name__ == "__main__":
    flask_app.run()
