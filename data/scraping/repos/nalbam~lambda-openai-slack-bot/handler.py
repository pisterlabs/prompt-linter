import boto3
import json
import os
import re
import sys
import time

# import deepl

from openai import OpenAI

from slack_bolt import App, Say
from slack_bolt.adapter.aws_lambda import SlackRequestHandler

BOT_CURSOR = os.environ.get("BOT_CURSOR", ":robot_face:")

# Keep track of conversation history by thread
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "openai-slack-bot-context")

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(DYNAMODB_TABLE_NAME)

# Set up Slack API credentials
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]

# Initialize Slack app
app = App(
    token=SLACK_BOT_TOKEN,
    signing_secret=SLACK_SIGNING_SECRET,
    process_before_response=True,
)

# Set up OpenAI API credentials
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

OPENAI_SYSTEM = os.environ.get("OPENAI_SYSTEM", "")
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", 0.5))

MESSAGE_MAX = int(os.environ.get("MESSAGE_MAX", 4000))

openai = OpenAI(
    api_key=OPENAI_API_KEY,
)

bot_id = app.client.api_call("auth.test")["user_id"]

# # Set up DeepL API credentials
# DEEPL_API_KEY = os.environ["DEEPL_API_KEY"]
# DEEPL_TARGET_LANG = os.environ.get("DEEPL_TARGET_LANG", "KR")


# Get the context from DynamoDB
def get_context(id, default=""):
    item = table.get_item(Key={"id": id}).get("Item")
    return (item["conversation"]) if item else (default)


# Put the context in DynamoDB
def put_context(id, conversation=""):
    expire_at = int(time.time()) + 86400  # 24 hours
    table.put_item(
        Item={
            "id": id,
            "conversation": conversation,
            "expire_at": expire_at,
        }
    )


# Update the message in Slack
def chat_update(channel, message, latest_ts):
    # print("chat_update: {}".format(message))
    app.client.chat_update(
        channel=channel,
        text=message,
        ts=latest_ts,
    )


# # Handle the translate test
# def translate(message, target_lang=DEEPL_TARGET_LANG, source_lang=None):
#     print("translate: {}".format(message))

#     translator = deepl.Translator(DEEPL_API_KEY)

#     result = translator.translate_text(message, target_lang=target_lang, source_lang=source_lang)

#     print("translate: {}".format(result))

#     return result


# Handle the openai conversation
def conversation(say: Say, thread_ts, prompt, channel, client_msg_id):
    print(thread_ts, prompt)

    # Keep track of the latest message timestamp
    result = say(text=BOT_CURSOR, thread_ts=thread_ts)
    latest_ts = result["ts"]

    messages = []

    # Add the user message to the conversation history
    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    if thread_ts != None:
        # Get thread messages using conversations.replies API method
        response = app.client.conversations_replies(channel=channel, ts=thread_ts)

        print("conversations_replies", response)

        if not response.get("ok"):
            print("Failed to retrieve thread messages")

        for message in response.get("messages", [])[::-1]:
            if message.get("client_msg_id", "") == client_msg_id:
                continue

            role = "user"
            if message.get("bot_id", "") != "":
                role = "assistant"

            content = message.get("text", "")

            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

            # print("messages size", sys.getsizeof(messages))

            if sys.getsizeof(messages) > MESSAGE_MAX:
                messages.pop(0)  # remove the oldest message
                break

    if OPENAI_SYSTEM != "":
        messages.append(
            {
                "role": "system",
                "content": OPENAI_SYSTEM,
            }
        )

    try:
        messages = messages[::-1]  # reversed

        print("messages", messages)
        print("messages size", sys.getsizeof(messages))

        stream = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=OPENAI_TEMPERATURE,
            stream=True,
        )

        # Stream each message in the response to the user in the same thread
        counter = 0
        message = ""
        for part in stream:
            if counter == 0:
                print("stream part", part)

            message = message + (part.choices[0].delta.content or "")

            # Send or update the message, depending on whether it's the first or subsequent messages
            if counter % 32 == 1:
                chat_update(channel, message + " " + BOT_CURSOR, latest_ts)

            counter = counter + 1

        # Send the final message
        chat_update(channel, message, latest_ts)

        print(thread_ts, message)

    except Exception as e:
        chat_update(channel, message, latest_ts)

        message = "Error handling message: {}".format(e)
        say(text=message, thread_ts=thread_ts)

        print(thread_ts, message)

        message = "Sorry, I could not process your request.\nhttps://status.openai.com"
        say(text=message, thread_ts=thread_ts)


# Handle the app_mention event
@app.event("app_mention")
def handle_mention(body: dict, say: Say):
    print("handle_mention: {}".format(body))

    event = body["event"]

    if "bot_id" in event:  # Ignore messages from the bot itself
        return

    thread_ts = event["thread_ts"] if "thread_ts" in event else event["ts"]
    prompt = re.sub(f"<@{bot_id}>", "", event["text"]).strip()
    channel = event["channel"]
    client_msg_id = event["client_msg_id"]

    conversation(say, thread_ts, prompt, channel, client_msg_id)


# Handle the DM (direct message) event
@app.event("message")
def handle_message(body: dict, say: Say):
    print("handle_message: {}".format(body))

    event = body["event"]

    if "bot_id" in event:  # Ignore messages from the bot itself
        return

    prompt = event["text"].strip()
    channel = event["channel"]
    client_msg_id = event["client_msg_id"]

    # Use thread_ts=None for regular messages, and user ID for DMs
    conversation(say, None, prompt, channel, client_msg_id)


# Handle the message event
def lambda_handler(event, context):
    body = json.loads(event["body"])

    if "challenge" in body:
        # Respond to the Slack Event Subscription Challenge
        return {
            "statusCode": 200,
            "headers": {"Content-type": "application/json"},
            "body": json.dumps({"challenge": body["challenge"]}),
        }

    print("lambda_handler: {}".format(body))

    # Duplicate execution prevention
    token = body["event"]["client_msg_id"]
    prompt = get_context(token)
    if prompt == "":
        put_context(token, body["event"]["text"])
    else:
        return {
            "statusCode": 200,
            "headers": {"Content-type": "application/json"},
            "body": json.dumps({"status": "Success"}),
        }

    # Handle the event
    slack_handler = SlackRequestHandler(app=app)
    return slack_handler.handle(event, context)
