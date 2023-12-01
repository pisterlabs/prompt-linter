import logging
from dotenv import load_dotenv
import requests
from langchain.llms import OpenAIChat
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from iblGpt.settings import SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET


from slack_bolt import App
from chatbot.models import MessageClient, MessageEntry

load_dotenv()

logger = logging.getLogger(__name__)

app = App(
    token=SLACK_BOT_TOKEN,
    signing_secret=SLACK_SIGNING_SECRET,
    # disable eagerly verifying the given SLACK_BOT_TOKEN value
    token_verification_enabled=False,
)


@app.event("app_mention")
def handle_app_mentions(logger, event, say):
    logger.info(event)
    say(f"Hi there, <@{event['user']}>")


@app.command("/mentor")
def send_mentor_message(ack, respond, command):
    # Acknowledge command request
    ack()
    message = command["text"]
    response = requests.post(
        "http://api.mentor.ibl.ai/ask/",
        json={"question": message, "database": "default", "with_sources": "true"},
    )
    body = response.json()
    respond(body["answer"])


@app.event("message")
def event_message(body, say, logger):
    logger.info(body)
    message = body["event"]["text"]
    docs = []
    if body["event"]["channel_type"] == "im":
        conversation_id = body["event"]["user"]
    else:
        conversation_id = body["event"]["channel"]
    client = MessageClient.objects.get_or_create(conversation_id=conversation_id)[0]
    history = client.get_history()
    if history:
        docs = [Document(page_content=history)]
    chain = load_qa_chain(OpenAIChat(model_name="gpt-4"), chain_type="stuff")
    try:
        result = chain(
            {"input_documents": docs, "question": message}, return_only_outputs=True
        )
    except:
        result = {"output_text": "We are very busy right now :( Please try again."}
    response = result["output_text"].strip()
    MessageEntry.objects.create(
        question=message, answer=result["output_text"], client=client
    )
    say(response)
