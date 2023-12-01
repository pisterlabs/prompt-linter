import logging

logging.basicConfig(level=logging.DEBUG, filename="chatter.log")
import asyncio
from lib.chatter import Chatter
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.starlette.async_handler import AsyncSlackRequestHandler
from models import ConfigRepository, PromptRepository

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route
import openai
from lib.schemas import Message
import re

slack_bot_token = asyncio.run(ConfigRepository.get_config_by_name("slack_bot_token"))
slack_signing_secret = asyncio.run(
    ConfigRepository.get_config_by_name("slack_signing_secret")
)
openai_key = asyncio.run(ConfigRepository.get_config_by_name("openapi_key"))
prompt_text = asyncio.run(PromptRepository.get_prompt_by_name("generic"))

chatter = Chatter(openai_key, prompt_text)

app = AsyncApp(
    token=slack_bot_token,
    signing_secret=slack_signing_secret,
    ignoring_self_events_enabled=False,
)

app_handler = AsyncSlackRequestHandler(app)


@app.middleware
async def log_request(logger, body, next):
    # middleware so that our bot can become aware of our user id, so that we can differentiate in the message event.
    # i believe the alternative to this hack is to go full on oauth to get access to the full suite of api calls, and that sounds awful.
    if not await chatter.get_id():
        await chatter.set_id(body["authorizations"][0]["user_id"])
    return await next()


# if someone mentions the app
@app.event("app_mention")
async def handle_app_mentions(body, say, logger):
    # if this mention is in a thread, and there's a convo attached to it, then bail, because we're gonna let
    # the message event capture this message.
    if "thread_ts" in body["event"]:
        chat_session = chatter.get_session(body["event"]["thread_ts"])
        if chat_session:
            return
    # else, this is a mention either in a thread somewhere, or in a chatter thread.
    # try to get a completion and then reply to the mention either in a thread, or top-level message.
    # if the mention also occurs in a thread started by chatter (someone replying to a chatter message),
    # then ignore. we'll handle it in the message event.
    try:
        chat_session = await chatter.new_session(body["event"]["ts"])
        response = await chat_session.chat(
            "user", re.sub(r"<@[A-Z0-9]+>", "", body["event"]["text"])
        )
    except openai.error.RateLimitError:
        response = "OpenAI is having problems, I can't respond right now :("
    logger.info(f"Memory: {chatter.memory.cache}")
    logger.info(f"Responding with {response.content}")
    if "thread_ts" in body["event"]:
        await say({"text": response.content, "thread_ts": body["event"]["thread_ts"]})
    else:
        await say(response.content)


@app.event("message")
async def handle_message_events(body, say, logger):
    # message event listens for all messages in a channel.
    # if a message comes from the bot's own user id, then we need to store the message in our LRU to maintain convo context.
    # check if it's a top level message, or a thread reply, and store accordingly.
    if body["event"]["user"] == await chatter.get_id():
        # if a top level chatter message comes our way, we need to remember it for future conversations.
        if "thread_ts" not in body["event"]:
            # it's a top level chatter message, we need to go ahead and remember this convo
            chatter.memory.push(
                body["event"]["ts"], Message("assistant", body["event"]["text"])
            )
            logger.info(f"Memory: {chatter.memory.cache}")

    else:
        # if it's a non-chatter reply to a thread, we need to fetch a completion and reply, and then store the response in the LRU.
        if "thread_ts" in body["event"]:
            # if chatter can remember the conversation, then build a session from it and then chat with it
            if chatter.memory.get(body["event"]["thread_ts"]):
                chat_session = await chatter.build_session(body["event"]["thread_ts"])
                if chat_session:
                    try:
                        response = await chat_session.chat(
                            "user", re.sub(r"<@[A-Z0-9]+>", "", body["event"]["text"])
                        )
                    except openai.error.RateLimitError:
                        response = (
                            "OpenAI is having problems, I can't respond right now :("
                        )
                    logger.info(f"Memory: {chatter.memory.cache}")
                    logger.info(f"Responding with {response.content}")
                    await say(
                        {
                            "text": response.content,
                            "thread_ts": body["event"]["thread_ts"],
                        }
                    )


async def endpoint(req: Request):
    return await app_handler.handle(req)


api = Starlette(
    debug=False, routes=[Route("/slack/events", endpoint=endpoint, methods=["POST"])]
)
