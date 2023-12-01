#do this
#pip install slack-sdk slack-bolt

import os, time, zipfile, csv, random, traceback
import logging, requests
from pathlib import Path
from datetime import datetime, timedelta
import openai, shodan
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# setup logging
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
pid = os.getpid()
logging.info(f"Process ID: {pid}")

# Set up your API keys
openai.api_key = ""
shodan_api_key = ""
SLACK_APP_TOKEN = "your_app_token"
SLACK_BOT_TOKEN = "your_bot_token"

shodan_api = shodan.Shodan(shodan_api_key)

app = App(token=SLACK_BOT_TOKEN)

last_interaction = datetime.now() - timedelta(hours=24)

instructions = """
Welcome to Root Bot, your AI assistant for exploring the world of hacking, security, and privacy.

Here are the available commands:
- !chat [message]: Chat with Root Bot and get answers to your questions.
- !shodan [query]: Search for devices using the Shodan API.
- !pwned [email]: Check if an email address has been involved in a data breach using the HIBP API.
- !history: View the history of commands used with Root Bot.
- !help: Display this message.
"""

# ... (rest of the code remains the same, except for the bot commands)

@app.event("app_mention")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ...
def command_handler(body, say):
    text = body['event'].get('text')
    if text:
        command = text.split(' ')[1]
        args = text.split(' ')[2:]
        if command == "!chat":
            asyncio.run(chat(ctx, message=" ".join(args)))
        elif command == "!shodan":
            asyncio.run(shodan_query(ctx, query=" ".join(args)))
        elif command == "!pwned":
            asyncio.run(pwned(ctx, email=" ".join(args)))
        elif command == "!history":
            asyncio.run(history(ctx))
        elif command == "!help":
            asyncio.run(help_command(ctx))

if __name__ == "__main__":
    if should_zip_history():
        zip_command_history()
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
