import os
import openai
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.webhook import WebhookClient
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Your tokens
SLACK_BOT_TOKEN = " "
SLACK_APP_TOKEN = " "
OPENAI_API_KEY  = " "
WEBHOOK_URL     = " "

# Create a WebClient instance
client = WebClient(token=SLACK_BOT_TOKEN)

# Create a WebhookClient instance
webhook_client = WebhookClient(WEBHOOK_URL)

# Event API & Web API
app = App(token=SLACK_BOT_TOKEN)

# This gets activated when the bot is tagged in a channel
@app.event("app_mention")
def handle_message_events(body, say):
    # Log message
    print(str(body["event"]["text"]).split(">")[1])

    # Create prompt for ChatGPT
    prompt = str(body["event"]["text"]).split(">")[1]

    # Let the user know that we are busy with the request
    try:
        webhook_client.send(
            text="Hello from your bot! :robot_face:\nThanks for your request, I'm on it!"
        )
    except SlackApiError as e:
        print(f"Error posting message to Slack: {e.response['error']}")

    # Check ChatGPT
    openai.api_key = OPENAI_API_KEY
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5
    ).choices[0].text

    # Reply to thread
    try:
        webhook_client.send(
            text=f"Here you go:\n{response}"
        )
    except SlackApiError as e:
        print(f"Error posting message to Slack: {e.response['error']}")

if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
