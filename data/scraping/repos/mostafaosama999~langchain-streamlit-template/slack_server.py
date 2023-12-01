import os
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Set Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]
EMBEDDINGS = OpenAIEmbeddings() # VECTOR SIZE IS DEPENDENT ON THIS, hf is 384

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the Flask app
# Flask is a web application framework written in Python
flask_app = Flask(__name__)
handler = SlackRequestHandler(app)
# initialize pinecone
pinecone.init(
    api_key=str(os.environ['PINECONE_API_KEY']),  
    environment=str(os.environ['PINECONE_ENV'])  
)
docsearch = Pinecone.from_existing_index(os.environ['PINECONE_INDEX_NAME'], EMBEDDINGS)

def query_similarity_search_QA_w_sources_OpenAI_Model(question):
    model = OpenAI(model_name="text-davinci-003")
    sources_chain = load_qa_with_sources_chain(model, chain_type="refine")
    result = sources_chain.run(input_documents=docsearch.similarity_search(question), question=question)
    return result

# Decorator for handling direct bot message events
@app.event("message")
def handle_direct_message(event, say):
    if event.get("subtype") is None and event.get("channel_type") == "im":
        user_input = event["text"]
        say("Hang on ... I am thinking ...")
        result = query_similarity_search_QA_w_sources_OpenAI_Model(user_input)
        say(result)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """
    Route for handling Slack events.
    This function passes the incoming HTTP request to the SlackRequestHandler for processing.

    Returns:
        Response: The result of handling the request.
    """
    return handler.handle(request)


# Run the Flask app
if __name__ == "__main__":
    flask_app.run(port=3000)