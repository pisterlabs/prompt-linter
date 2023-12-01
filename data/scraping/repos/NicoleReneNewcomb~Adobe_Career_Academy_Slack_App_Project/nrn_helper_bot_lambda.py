""" Nicole-Rene's Helper Bot Slack App for Lambda
    Created by: Nicole-Rene Newcomb
    Newest Version Date: 08/01/2023
    Description: This Slack App Bot operates via AWS Lambda
"""

import logging
import random
import re
import os
import openai
from cute_animal_photos import cute_animals_photos
from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler

# Setting logging details and formatting message output
SlackRequestHandler.clear_all_log_handlers()
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)

# Retrieve tokens from environmental variables
SLACK_BOT_TOKEN = os.environ['SLACK_BOT_TOKEN']
SLACK_SIGNING_SECRET = os.environ['SLACK_SIGNING_SECRET']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# Parameter provides rapid response to requests (avoids timeout when using FaaS)
nrn_helper_bot = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET, ignoring_self_events_enabled=True, process_before_response=True)

# Main handler for incoming requests to AWS Lambda function from Slack
def lambda_handler(event, context):
    """handles incoming requests from Slack to AWS Lambda function"""
    # Create instance of bot
    slack_handler = SlackRequestHandler(app=nrn_helper_bot)
    # Route request to proper handler
    return slack_handler.handle(event, context)
    

# Provides acknowledgement to Slack for longer functions
def ack_3s_response(ack):
    ack()



########### Event/command handlers section ###########


# Function to process app_mention: @Nicole-Rene's Helper Bot
def mention_handler(body, event, say, logger):
    """handles app_mention events when @<botname> referenced"""
    logger.info(body)
    user = event['user']
    bot_help_txt = "\nPlease enter /help to see available commands."
    bot_message = f"Hello there, <@{user}>! How can I help you? {bot_help_txt}"
    say(text=bot_message)

# Event handler for when Nicole-Rene's Helper Bot is mentioned by name
nrn_helper_bot.event("app_mention")(ack=ack_3s_response, lazy=[mention_handler])


# Function to process message including ":wave:" emoji
def wave_emoji(message, say):
    """handles a wave emoji message"""
    user = message['user']
    say(text=f"And a :wave: to you too, <@{user}>!")

# Event handler to listen for the ":wave:" emoji in messages
nrn_helper_bot.message(":wave:")(ack=ack_3s_response, lazy=[wave_emoji])


# Function to process message containing negative emotion keywords
def triggerred_inspiration_dose(message, say):
    """handles response to overhearing negative messages"""
    user = message['user']
    user_message = message['text']
    openai.api_key = OPENAI_API_KEY
    prompt_message = f"Please create a inspirational quote to motivate someone who just said the following: {user_message}"
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "user", "content": prompt_message}
        ]
    )
    text_reply = response["choices"][0]["message"]["content"]
    say(text=f"Take heart, <@{user}>! Here's an inspirational quote from ChatGPT:\n\n{text_reply}")

# Event handler to listen for messages with negative emotions that need inspiration/encouragement
nrn_helper_bot.message(re.compile("(annoyed|angry|depressed|frustrated|mad|sad|unhappy|upset)"))(ack=ack_3s_response, lazy=[triggerred_inspiration_dose])


# Event handler to listen for general message events
@nrn_helper_bot.event("message")
def message_response(ack):
    """handles general message events from channel"""
    #Acknowledges receipt of event, but no content sent to channel
    ack()


# Function to process /help command
def help_command(say):
    """handles /help command"""
    option1 = "\n/help - displays the available commands\n/cuteanimals - displays a random photo of cute animals"
    option2 = "\n/inspiration - displays a random famous inspirational quote\n/chatgpt + message/prompt - requests a response from ChatGPT"
    option3 = "\nMention @Nicole-Rene's Helper Bot - bot greets user and suggests /help command"
    option4 = "\nMention :wave: in message - bot waves back to you\nMention feeling a negative emotion - ChatGPT responds with an encouraging message"

    bot_message = f"\n \n \nHere are some options: {option1}{option2}{option3}{option4}"
    say(text=bot_message)

# Command handler to respond to /help command
nrn_helper_bot.command("/help")(ack=ack_3s_response, lazy=[help_command])


# Function to process /chatgpt command
def chatgpt_request(body, say):
    """handles /chatgpt command"""
    openai.api_key = OPENAI_API_KEY
    prompt_message = body['text']
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "user", "content": prompt_message}
        ]
    )
    text_reply = response["choices"][0]["message"]["content"]
    say(text=f"\n \n \nHeres the response from ChatGPT to the prompt \"{prompt_message}\":\n\n{text_reply}")

# Event handler to listen for /chatgpt command
nrn_helper_bot.command("/chatgpt")(ack=ack_3s_response, lazy=[chatgpt_request])


# Function to process /inspiration command
def chatgpt_inspiration(say):
    """handles /inspiration command"""
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "user", "content": "Selecting a quote randomly to avoid repeating an answer given in the last 2 hours using this API key, please provide a famous inspirational or motivational quote along with the name of the speaker or author."}
        ]
    )
    text_reply = response["choices"][0]["message"]["content"]
    say(text=f"\n \n \nHeres your inspirational message from ChatGPT:\n\n{text_reply}")

# Event handler to listen for /inspiration command
nrn_helper_bot.command("/inspiration")(ack=ack_3s_response, lazy=[chatgpt_inspiration])


# Function to process /cuteanimals command
def cute_animals(ack, say):
    """responding to a wave emoji message"""
    random_cute_animal = random.choice(cute_animals_photos)

    # Sends both a text message and an image url back to Slack
    say(
        {
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "\n \n \nHere's a cute animal photo for you:",
                    },
                },
                {
                    "type": "image",
                    "title": {
                        "type": "plain_text",
                        "text": "Cute Animal",
                    },
                    "image_url": random_cute_animal,
                    "alt_text": "Cute Animal",
                }
            ]
        }
    )

# Event handler to respond to /cuteanimals command
nrn_helper_bot.command("/cuteanimals")(ack=ack_3s_response, lazy=[cute_animals])



######## End of event/command handlers section ########
