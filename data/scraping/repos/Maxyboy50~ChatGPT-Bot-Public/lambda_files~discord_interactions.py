import requests
import asyncio
import aiohttp
import openai
import os
from dynamo_db_interactions import fetch_message_history
from chat_gpt_interactions import chat_gpt_message
from discord import Webhook


API_KEY = os.getenv("API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
openai.api_key = API_KEY


async def send_webhook(message):
    async with aiohttp.ClientSession() as session:
        max_characters = 2000
        webhook = Webhook.from_url(WEBHOOK_URL, session=session)
        if len(message) <= max_characters:
            await webhook.send(message)
        else:
            chunks = [
                message[i : i + max_characters]
                for i in range(0, len(message), max_characters)
            ]
            for chunk in chunks:
                await webhook.send(chunk)


def defer(interaction_id: str, interaction_token: str):
    url = f"https://discord.com/api/v10/interactions/{interaction_id}/{interaction_token}/callback"
    response = {"type": 5}
    headers = {"Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=response)


def follow_up(response: str, application_id: str, interaction_token: str):
    url = f"https://discord.com/api/v10/webhooks/{application_id}/{interaction_token}"
    headers = {"Content-Type": "application/json"}
    response = {"content": f"{response}"}
    r = requests.post(url, json=response)
    if "BASE_TYPE_MAX_LENGTH" in r.text:
        too_long_payload = {
            "content": "The prompt/response was too long to be returned to Discord. Think of a shorter prompt and try again."
        }
        r = requests.post(url, headers=headers, json=too_long_payload)


def command_handler(
    command_name: str,
    interaction_token: str,
    userID: str,
    interaction_id: str,
    dynamo_db_table: None,
    application_id: str,
    command_value: str,
):
    try:
        defer(interaction_id=interaction_id, interaction_token=interaction_token)
        conversation_history = fetch_message_history(
            userID=userID, table=dynamo_db_table
        )
        if command_name == "chatgptprompt":
            assistant_response = chat_gpt_message(
                messages=conversation_history,
                prompt=command_value,
                userID=userID,
                table=dynamo_db_table,
                max_tokens=325,
            )
            follow_up(
                response=assistant_response,
                application_id=application_id,
                interaction_token=interaction_token,
            )
        elif command_name == "longmessage":
            follow_up(
                response="Consulting my assistant, expect a response soon!",
                application_id=application_id,
                interaction_token=interaction_token,
            )
            assistant_response = chat_gpt_message(
                messages=conversation_history,
                prompt=command_value,
                userID=userID,
                table=dynamo_db_table,
                max_tokens=2000,
            )
            asyncio.run(send_webhook(assistant_response))
        else:
            follow_up(
                response="Sorry, that is an invalid command/option",
                application_id=application_id,
                interaction_token=interaction_token,
            )
    except openai.error.RateLimitError:
        follow_up(
            response="Sorry, the OpenAI servers are overloaded right now, please try again shortly",
            application_id=application_id,
            interaction_token=interaction_token,
        )
