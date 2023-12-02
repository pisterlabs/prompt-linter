#==================================================================
# Created by Davide Pasca - 2023/09/14
#==================================================================

# Running the bot:
# 1. Run "ngrok http 127.0.0.1:5000" to expose the local server to the Internet
# 2. Copy the ngrok URL and:
#     - Go to console.twilio.com : Messaging -> Try it out -> Send a WhatsApp message
#     - Set <ngrok URL>/whatsapp to "When a message comes in"
# 3. Run "python wa_bot.py" to start the Flask server

import os
from flask import Flask, request, Response
from twilio.twiml.messaging_response import MessagingResponse
import openai

SYSTEM_PROMPT_CHARACTER = (
    "You are a skillful highly logical assistant that goes straight to the point, "
    "with a tiny bit of occasional sarcasm."
)

SYSTEM_PROMPT_FIXED_FORMAT = (
    "You are operating in a forum, where multiple users can interact with you. "
)

# Initialize Flask app
app = Flask(__name__)
# Read OPENAI_API_KEY from environment variables
openai.api_key = os.environ.get("OPENAI_API_KEY")

@app.route('/whatsapp', methods=['POST'])
def whatsapp_bot():
    print(request.headers)
    # Get the incoming message
    incoming_msg = request.values.get('Body', '')
    print(f"Received message: {incoming_msg}")

    user_number = request.values.get('From', '')
    print(f"From: {user_number}")

    # Initialize response object
    resp = MessagingResponse()
    msg = resp.message()

    # Your OpenAI logic here
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT_CHARACTER + SYSTEM_PROMPT_FIXED_FORMAT},
        {"role": "user", "content": incoming_msg}
    ]

    print(f"Conversation: {conversation}")

    try:
        openai_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=conversation
        )
    except Exception as e:
        print(f"OpenAI API Error: {e}")

    reply_text = openai_response['choices'][0]['message']['content']
    print(f"Reply: {reply_text}")

    # Respond to the message
    msg.body(reply_text)

    return str(resp)

if __name__ == '__main__':
    app.run(debug=True)
