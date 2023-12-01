import copy
import os
import re

from threading import Thread
from time import sleep
from typing import Dict
from typing import List

import openai

from dotenv import load_dotenv
from flask import Flask
from flask import request
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse

from genai.eyfs.eyfs import TextGenerator
from genai.utils import read_json


load_dotenv()
# Twilio settings
client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
# OpenAI model
LLM = "gpt-3.5-turbo"
TEMPERATURE = 0.5
openai.api_key = os.environ["OPENAI_API_KEY"]

# Prepare ELI3 base prompt
ELI3_MESSAGES = read_json("src/genai/whatsapp_bot/prompts/eli3/eli3.json")

# Prepare EYFS base prompt and parameters
areas_of_learning = [
    "Communication and Language",
    "Personal, Social and Emotional Development",
    "Physical Development",
    "Literacy",
    "Mathematics",
    "Understanding the World",
    "Expressive Arts and Design",
]
areas_of_learning_desc = read_json("src/genai/eyfs/areas_of_learning.json")
areas_of_learning_text = [v for k, v in areas_of_learning_desc.items() if k in areas_of_learning]
eyfs_paths = [
    "src/genai/eyfs/prompts/system.json",
    "src/genai/eyfs/prompts/context_and_task.json",
    "src/genai/eyfs/prompts/constraints.json",
    "src/genai/eyfs/prompts/situation.json",
]
EYFS_MESSAGES = [read_json(path) for path in eyfs_paths]
EYFS_PARAMETERS = {
    "areas_of_learning": areas_of_learning,
    "n_results": 5,
    "location": "Indoors or Outdoors",
    "areas_of_learning_text": areas_of_learning_text,
}

# Initiate the Flask app
app = Flask(__name__)


def format_activities_text(text: str) -> str:
    """Format the response from the EYFS generator for better display in WhatsApp"""
    text = (
        text.replace("## Conversations", "*Conversations*\n")
        .replace("## Games and Crafts", "*Games and Crafts*\n")
        .replace("**Activity description**", "_Activity description_")
        .replace("**Areas of learning**", "_Areas of learning_")
    )
    # replace markdown subheadings with bold italics
    text = re.sub(r"###\s*(.+)", r"*_\1_*", text)
    return text


def generate_reply(incoming_message: str, sender_contact: str, receiver_contact: str) -> str:
    """Parse message text and return an appropriate response.

    Presently supports two types of responses: 'explain' and 'activities'
    Activities response is threaded to allow for longer response times. This is a very basic
    workaround to the 15 second timeout limit imposed by Twilio.

    Args:
        incoming_message:
            Message text
        sender_contact:
            Sender's contact, follows a format 'whatsapp:+<phone number>'
        receiver_contact:
            Receiver's contact (ie, my contact), follows a format 'whatsapp:+<phone number>'

    Returns:
        Response text
    """
    text_message = incoming_message.lower()

    # 'explain' response
    if text_message[0:7] == "explain":
        response = TextGenerator.generate(
            model=LLM,
            temperature=TEMPERATURE,
            messages=[ELI3_MESSAGES.copy()],
            message_kwargs={"input": text_message[7:].strip()},
        )
        return response["choices"][0]["message"]["content"]
    # 'activities' response
    elif "activities" in text_message[0:10]:
        EYFS_PARAMETERS["description"] = text_message
        thread = Thread(
            target=send_text, args=[copy.deepcopy(EYFS_MESSAGES), EYFS_PARAMETERS, receiver_contact, sender_contact]
        )
        thread.start()
        return "Thank you for your question. I am thinking..."
    else:
        # Return a default message
        return (
            'Write "Explain <your question>" to explain a concept to a 3-year old \n\n or'
            + '\n\n "Activities <your topic>" to get activity ideas'
        )


def send_text(messages: List[Dict], message_kwargs: Dict, my_contact: str, receiver_contact: str) -> None:
    """Generate text messages and send them to a given contact

    Args:
        messages:
            List of messages to be used as prompts
        message_kwargs:
            Dictionary of keyword arguments to be passed to the TextGenerator
        my_contact:
            Sender's contact, follows a format 'whatsapp:+<phone number>'
        receiver_contact:
            Receiver's contact (ie, my contact), follows a format 'whatsapp:+<phone number>'
    """
    # Generate response to the message
    response = TextGenerator.generate(
        model=LLM,
        temperature=TEMPERATURE,
        messages=messages,
        message_kwargs=message_kwargs,
    )
    text_body = response["choices"][0]["message"]["content"]
    # Format the text_body for better display on WhatsApp
    text_body = format_activities_text(text_body)
    # Divide output into 1500 character chunks due to WhatsApp character limit of 1600 chars
    texts = [text_body[i : i + 1500] for i in range(0, len(text_body), 1500)]
    # Send message
    for text in texts:
        client.messages.create(body=text, from_=my_contact, to=receiver_contact)
        sleep(0.5)
    return


@app.route("/")
def hello_world() -> str:
    """Information message"""
    return "Nesta generative AI prototype: WhatsApp bot for suggesting kids activities"


@app.route("/text", methods=["POST"])
def text_reply() -> str:
    """Respond to incoming messages"""
    reply = generate_reply(
        incoming_message=request.form.get("Body"),
        sender_contact=request.form.get("From"),
        receiver_contact=request.form.get("To"),
    )
    resp = MessagingResponse()
    resp.message(reply)
    return str(resp)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
