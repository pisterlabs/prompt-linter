import openai
import logging

from openai import OpenAIError
from Dataclasses import ConversationEntry, ConversationStatus


class OpenAI:
    def __init__(
        self,
        org,
        key,
        prompt_message,
        thinking_message,
        error_message,
        memory_size=10,
        chat_wide_conversation=False,
    ) -> None:
        openai.organization = org
        openai.api_key = key
        self.prompt = prompt_message
        self.thinking = thinking_message
        self.error = error_message
        self.tokens = 0
        self.conversations = {}
        self.conversations_status = {}
        self.memory_size = int(memory_size)
        self.chat_wide_conversation = chat_wide_conversation

    def start_conversation(self, conversation_id, author):
        self.conversations[conversation_id] = [
            ConversationEntry(
                "system",
                self.prompt.format(username=author),
                "Twitch",
            )
        ]
        self.conversations_status[conversation_id] = ConversationStatus.IDLE

    def reprompt_conversation(
        self, conversation_id, prompt: str = None, author: str = None
    ):
        self.clean_conversation(conversation_id)
        conversation_prompt = (
            prompt if prompt else self.prompt.format(username=author)
        )
        self.conversations[conversation_id] = [
            ConversationEntry("system", conversation_prompt, author)
        ]
        self.conversations_status[conversation_id] = ConversationStatus.IDLE

    def clean_conversation(self, conversation_id):
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
        if conversation_id in self.conversations_status:
            del self.conversations_status[conversation_id]

    def add_message(self, conversation_id, role, message, author):
        self.conversations[conversation_id].append(
            ConversationEntry(role, f"{message}", author)
        )
        if len(self.conversations[conversation_id]) > self.memory_size:
            del self.conversations[conversation_id][1:3]

    def get_conversation(self, conversation_id):
        logging.debug(self.conversations[conversation_id])
        return self.conversations[conversation_id]

    def get_conversations_status(self, conversation_id, author):
        if conversation_id not in self.conversations_status:
            self.start_conversation(conversation_id, author)
        logging.debug(
            f"Conversation status for {conversation_id} is {self.conversations_status[conversation_id]}"
        )
        return self.conversations_status[conversation_id]

    def set_conversations_status(self, conversation_id, status):
        self.conversations_status[conversation_id] = status

    async def request_chat(self, messages, assistant_message=None):
        """
        $0.0015 per 1000 tokens using gpt-3.5-turbo-0613
        Which is 1/10th of the cost of text-davinci-003
        Meaning that even with a larger prompt, this is still cheaper
        """
        try:
            json_messages = [message.__dict__ for message in messages]
            if assistant_message:
                json_messages.append(assistant_message.__dict__)
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo-0613",
                messages=json_messages,
            )
            logging.info(response)
            return response
        except OpenAIError as e:
            logging.error(e)
            return False

    async def chat(
        self, username: str = None, message: str = None, channel: str = None
    ):
        author = username
        if self.chat_wide_conversation:
            conversation_id = f"{channel}__chat"
        else:
            conversation_id = f"{username}"
        if (
            self.get_conversations_status(conversation_id, author)
            == ConversationStatus.IDLE
        ):
            self.set_conversations_status(
                username, ConversationStatus.OCCUPIED
            )
            self.add_message(conversation_id, "user", message, author)
            assistant_message = (
                ConversationEntry(
                    "assistant",
                    f"Please respond to @{author}'s last message: '{message}'. "
                    "Consider the context and adress them directly.",
                    "Twitch",
                )
                if self.chat_wide_conversation
                else None
            )
            response = await self.request_chat(
                self.get_conversation(conversation_id), assistant_message
            )
            if response:
                reply = response["choices"][0]["message"]["content"]
                self.add_message(
                    conversation_id, "assistant", reply, "botdelicious"
                )
            else:
                reply = self.error.format(username=username)
            self.set_conversations_status(
                conversation_id, ConversationStatus.IDLE
            )
        else:
            reply = self.thinking.format(username=username)
        return reply

    async def shoutout(
        self, target: dict = None, author: str = None, failed: bool = False
    ) -> str:
        system_name = "ai_shoutout_generator"
        system_prompt = "Hype Twitch Streamer Shoutout Generator"

        if failed:
            system_message = (
                f"Give a snarky reply about how @{author} "
                f"tried to shoutout @{failed}, but that user doesn't exist."
            )
        else:
            live_message = (
                "is currently live and is"
                if target["is_live"]
                else "is currently not live, but was last seen"
            )
            system_message = (
                f"Write a shoutout for a Twitch streamer named "
                f"{target['display_name']} who {live_message} "
                f"playing {target['game_name']} with the "
                f"stream title {target['title']}. "
                f"This is their description: {target['description']}. "
                f"These are their tags: "
                f"{', '.join([str(tag) for tag in target['tags']])}. "
                f"Do not list the tags in the reply. "
                f"Make sure to end the reply with their url: "
                f"https://twitch.tv/{target['name']}. "
                f"Keep the reply under 490 characters."
            )

        self.reprompt_conversation(
            system_name, prompt=system_prompt, author="Twitch"
        )
        self.add_message(system_name, "user", {system_message}, author)
        response = await self.request_chat(self.get_conversation(system_name))
        reply = response["choices"][0]["message"]["content"]

        return reply
