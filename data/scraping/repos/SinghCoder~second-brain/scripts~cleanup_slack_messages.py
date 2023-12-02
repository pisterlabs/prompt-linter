import os

from dotenv import load_dotenv
from langchain.tools import tool
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

load_dotenv()


# Initialize the Slack WebClient with your bot token
slack_token = os.environ["SLACK_BOT_TOKEN"]  # Replace with your bot token
client = WebClient(token=slack_token)

# Define the user's ID and conversation ID
user_id = os.environ.get("SLACK_USER_ID")  # Replace with the user's ID
conversation_id = "CONVERSATION_ID"  # Replace with the conversation ID

# List messages from the conversation
try:
    response = client.conversations_history(channel="D05N6RX8MAT")
    messages = response["messages"]

    # Delete messages sent by the bot
    for message in messages:
        print(message)
        # if message.get("user") == user_id:
        client.chat_delete(channel="D05N6RX8MAT", ts=message["ts"])

except SlackApiError as e:
    print("Error listing or deleting messages:", e.response["error"])
