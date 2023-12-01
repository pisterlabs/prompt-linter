#!/usr/bin/env python3

# Make this a python module:


# TODO: How is logging normally controlled?
import logging

logger = logging.getLogger(__name__)

import os
import re
from typing import List

from langchain import OpenAI
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.errors import SlackApiError

from .ConversationAI import ConversationAI

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get('SLACK_APP_TOKEN')
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")


class SlackBot:
    def __init__(self, slack_app: AsyncApp):
        self.threads_bot_is_participating_in = {}
        self.app = slack_app
        self.client = self.app.client
        self.id_to_name_cache = {}
        self.user_id_to_info_cache = {}

    async def start(self):
        logger.info("Looking up bot user_id. (If this fails, something is wrong with the auth)")
        response = await self.app.client.auth_test()
        self.bot_user_id = response["user_id"]
        self.bot_user_name = await self.get_username_for_user_id(self.bot_user_id)
        logger.info("Bot user id: "+ self.bot_user_id)
        logger.info("Bot user name: "+ self.bot_user_name)

        await AsyncSocketModeHandler(app, SLACK_APP_TOKEN).start_async()

    async def get_user_info_for_user_id(self, user_id):
        user_info = self.user_id_to_info_cache.get(user_id, None)
        if user_info is not None:
            return user_info
        
        user_info_response = await self.app.client.users_info(user=user_id)
        user_info = user_info_response['user']
        logger.debug(user_info)
        self.user_id_to_info_cache[user_id] = user_info
        return user_info

    async def get_username_for_user_id(self, user_id):
        user_info = await self.get_user_info_for_user_id(user_id)
        profile = user_info['profile']
        if (user_info['is_bot']):
            ret_val = profile['real_name']
        else:
            ret_val = profile['display_name']

        return ret_val
        
    async def upload_snippets(self, channel_id: str, thread_ts: str, response: str) -> str:
        matches: List[str] = re.findall(r"```(.*?)```", response, re.DOTALL)
        counter: int = 1
        for match in matches:
            match = match.strip()
            first_line: str = match.splitlines()[0]
            first_word: str = first_line.split()[0]
            extension: str = "txt"
            if first_word == "python":
                extension = "py"
            elif first_word in ["javascript", "typescript"]:
                extension = "js"
            elif first_word == "bash":
                extension = "sh"
            if not extension:
                if first_word:
                    extension = first_word
                else:
                    extension = "txt"
            file_response = await self.client.files_upload(channels=channel_id, content=match, filename=f"snippet_{counter}.{extension}", thread_ts=thread_ts)
            file_id: str = file_response["file"]["id"]
            response += "\n"+f"<https://slack.com/files/{self.bot_user_id}/{file_id}|code.{extension}>"
            counter += 1
        return response

    async def reply_to_slack(self, channel_id, thread_ts, message_ts, response):
        # In the future, we could take out any triple backticks code like:
        # ```python
        # print("Hello world!")
        # ```
        # And we could upload it to Slack as a file and then link to it in the response.
        # Let's try something - if they have an emoji, and only an emoji, in the response, let's react to the message with that emoji:
        # regex for slack emoji to ensure that the _entire_ message only consists of a single emoji:
        slack_emoji_regex = r"^:[a-z0-9_+-]+:$"
        if re.match(slack_emoji_regex, response.strip()):
            try:
                emoji_name=response.strip().replace(":", "")
                logger.info("Responding with single emoji: "+emoji_name)
                await self.client.reactions_add(channel=channel_id, name=emoji_name, timestamp=message_ts)
            except Exception as e:
                logger.exception(e)
            return
        else:
            await self.client.chat_postMessage(channel=channel_id, text=response, thread_ts=thread_ts)

    async def confirm_message_received(self, channel, thread_ts, message_ts, user_id_of_sender):
        # React to the message with a thinking face emoji:
        try:
            await self.client.reactions_add(channel=channel, name="thinking_face", timestamp=message_ts)
        except Exception as e:
            logger.exception(e)

    async def confirm_wont_respond_to_message(self, channel, thread_ts, message_ts, user_id_of_sender):
        # React to the message with a speak_no_evil emoji:
        try:
            await self.client.reactions_add(channel=channel, name="speak_no_evil", timestamp=message_ts)
        except Exception as e:
            logger.exception(e)


    async def respond_to_message(self, channel_id, thread_ts, message_ts, user_id, text):
        try:
            conversation_ai: ConversationAI = self.threads_bot_is_participating_in.get(thread_ts, None)
            if conversation_ai is None:
                raise Exception("No AI found for thread_ts")
            text = await self.translate_mentions_to_names(text)
            sender_user_info = await self.get_user_info_for_user_id(user_id)
            response = await conversation_ai.respond(sender_user_info, channel_id, thread_ts, message_ts, text)
            if (response is None):
                # Let's just put an emoji on the message to say we aren't responding
                await self.confirm_wont_respond_to_message(channel_id, thread_ts, message_ts, user_id)
            # We don't respond here since the bot is streaming responses
        except Exception as e:
            response = f":exclamation::exclamation::exclamation: Error: {e}"
            # Print a red error to the console:
            logger.exception(response)
            await self.reply_to_slack(channel_id, thread_ts, message_ts, response)

    @staticmethod
    def is_parent_thread_message(message_ts, thread_ts):
        return message_ts == thread_ts

    async def translate_mentions_to_names(self, text):
        # Replace every @mention of a user id with their actual name:
        # First, use a regex to find @mentions that look like <@U123456789>:
        matches = re.findall(r"<@(U[A-Z0-9]+)>", text)
        for match in matches:
            mention_string = f"<@{match}>"
            mention_name = await self.get_username_for_user_id(match)
            if mention_name is not None:
                text = text.replace(mention_string, "@"+mention_name)

        return text

    async def add_ai_to_thread(self, channel_id, thread_ts, message_ts):
        if thread_ts in self.threads_bot_is_participating_in:
            return

        processed_history = None
        # Is this thread_ts the very first message in the thread? If so, we need to create a new AI for it.
        if not self.is_parent_thread_message(message_ts, thread_ts):
            logger.debug("It looks like I am not the first message in the thread. I should get the full thread history from Slack and add it to my memory.")
            # This is not the very first message in the thread
            # We should figure out a way to boostrap the memory:
            # Get the full thread history from Slack:
            thread_history = await client.conversations_replies(channel=channel_id, ts=thread_ts)
            # Iterate through the thread history, adding each of these to the ai_memory:
            processed_history = []
            message_history = thread_history.data['messages']
            # Get rid of the last message from the histroy since it's the message we're responding to:
            message_history = message_history[:-1]
            for message in message_history:
                text = message['text']
                text = await self.translate_mentions_to_names(text)
                user_id = message['user']
                user_name = await self.get_username_for_user_id(user_id)
                if (user_id == self.bot_user_id):
                    processed_history.append({"bot": text})
                else:
                    # Get the username for this user_id:
                    processed_history.append({f"{user_name}": text})

        ai = ConversationAI(self.bot_user_name, self.client, processed_history)
        self.threads_bot_is_participating_in[thread_ts] = ai

    def is_ai_participating_in_thread(self, thread_ts, message_ts):
        if thread_ts in self.threads_bot_is_participating_in:
            return True
        return False

    def is_bot_mentioned(self, text):
        return f"<@{self.bot_user_id}>" in text

    async def on_message(self, event, say):
        message_ts = event['ts']
        thread_ts = event.get('thread_ts', message_ts)
        try:
            # {'client_msg_id': '7e605650-8b39-4f61-99c5-795a1168fb7c', 'type': 'message', 'text': 'Hi there Chatterbot', 'user': 'U024LBTMX', 'ts': '1679289332.087509', 'blocks': [{'type': 'rich_text', 'block_id': 'ins/', 'elements': [{'type': 'rich_text_section', 'elements': [{'type': 'text', 'text': 'Hi there Chatterbot'}]}]}], 'team': 'T024LBTMV', 'channel': 'D04V265MYEM', 'event_ts': '1679289332.087509', 'channel_type': 'im'}

            logger.info(f"Received message event: {event}")
            # At first I thought we weren't told about our own messages, but I don't think that's true. Let's make sure we aren't hearing about our own:
            if event.get('user', None) == self.bot_user_id:
                logger.debug("Not handling message event since I sent the message.")
                return

            start_participating_if_not_already = False
            channel_id = event['channel']
            # Is this message part of an im?
            channel_type = event.get('channel_type', None)
            if channel_type and channel_type == "im":
                # This is a direct message. So of course we should be participating if we are not
                start_participating_if_not_already = True
            # else if this is a message in a channel:
            elif self.is_bot_mentioned(event['text']):
                # This is a message in a channel, but it mentions us. So we should be participating if we are not
                start_participating_if_not_already = True

            if start_participating_if_not_already:
                await self.add_ai_to_thread(channel_id, thread_ts, message_ts)

            # And now, are we participating in it?
            if self.is_ai_participating_in_thread(thread_ts, message_ts):
                user_id = event['user']
                text = event['text']
                await self.confirm_message_received(channel_id, thread_ts, message_ts, user_id)
                await self.respond_to_message(channel_id, thread_ts, message_ts, user_id, text)
        except Exception as e:
            response = f":exclamation::exclamation::exclamation: Error: {e}"
            logger.exception(response)
            await say(text=response, thread_ts=thread_ts)

    async def on_member_joined_channel(self, event_data):
        # Get user ID and channel ID from event data
        user_id = event_data["user"]
        channel_id = event_data["channel"]

        user_info = await self.get_user_info_for_user_id(user_id)
        username = await self.get_username_for_user_id(user_id)
        profile = user_info.get("profile", {})
        llm_gpt3_turbo = OpenAI(temperature=1, model_name="gpt-3.5-turbo", request_timeout=30, max_retries=5, verbose=True)

        # TODO: Extract into yaml file instead:
        welcome_message = (await llm_gpt3_turbo.agenerate([f"""
You are a funny and creative slackbot {self.bot_user_name}
Someone just joined a Slack channel you are a member of, and you want to welcome them creatively and in a way that will make them feel special.
You are VERY EXCITED about someone joining the channel, and you want to convey that!
Their username is {username}, but when you mention their username, you should say "<@{user_id}>" instead.
Their title is: {profile.get("title")}
Their current status: "{profile.get("status_emoji")} {profile.get("status_text")}"
Write a slack message, formatted in Slack markdown, that encourages everyone to welcome them to the channel excitedly.
Use emojis. Maybe write a song. Maybe a poem.

Afterwards, tell the user that you look forward to "chatting" with them, and tell them that they can just mention <@{self.bot_user_id}> whenever they want to talk.
"""])).generations[0][0].text
        if welcome_message:
          try:
            # Send a welcome message to the user
            await self.client.chat_postMessage(channel=channel_id, text=welcome_message)
          except e:
            logger.exception("Error sending welcome message")


app = AsyncApp(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
client = app.client
slack_bot = SlackBot(app)

@app.event("message")
async def on_message(payload, say):
    logger.info("Processing message...")
    await slack_bot.on_message(payload, say)

# Define event handler for user joining a channel
@app.event("member_joined_channel")
async def handle_member_joined_channel(event_data):
    logger.info("Processing member_joined_channel event", event_data)
    await slack_bot.on_member_joined_channel(event_data)

@app.event('reaction_added')
async def on_reaction_added(payload):
    logger.info("Ignoring reaction_added")

@app.event('reaction_removed')
async def on_reaction_removed(payload):
    logger.info("Ignoring reaction_removed")

@app.event('app_mention')
async def on_app_mention(payload, say):
    logger.info("Ignoring app_mention in favor of handling it via the message handler...")

# This will take too long - maybe at some point we can do this and save it off somewhere
async def getCommonCustomEmojis(num_to_retrieve: int = 40):
    # Call the emoji.list API method to retrieve the list of custom emojis
    try:
        response = await client.emoji_list()
        emoji_list = response["emoji"]
    except SlackApiError as e:
        print("Error retrieving custom emojis: {}".format(e))

    # Retrieve the usage count for each custom emoji
    emoji_counts = []
    for emoji in emoji_list:
        try:
            response = await client.emoji_get(name=emoji)
            count = response["emoji"]["usage_count"]
            emoji_counts.append((emoji, count))
        except SlackApiError as e:
            print("Error retrieving usage count for emoji {}: {}".format(emoji, e))

    # Sort the list of custom emojis by usage count in descending order
    emoji_counts.sort(key=lambda x: x[1], reverse=True)

    # Retrieve the num_to_retrieve most commonly used custom emojis
    most_common_emojis = [emoji for emoji, count in emoji_counts[:num_to_retrieve]]

    return most_common_emojis