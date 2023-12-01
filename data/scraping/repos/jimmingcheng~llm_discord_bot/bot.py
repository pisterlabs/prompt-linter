from typing import List
from typing import Optional

import arrow
import re
from abc import ABC
from abc import abstractmethod
from discord import ChannelType
from discord import Message
from discord.ext import commands
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import ChatMessage

from llm_task_handler.chatroom import ChatroomConversation
from llm_task_handler.chatroom import ChatroomMessage
from llm_task_handler.dispatch import TaskDispatcher


def get_max_tokens(model_name: str):
    if model_name == 'gpt-4':
        max_tokens = 8192
    elif model_name == 'gpt-4-32k':
        max_tokens = 32768
    elif model_name == 'gpt-3.5-turbo':
        max_tokens = 4096
    elif model_name == 'gpt-3.5-turbo-16k':
        max_tokens = 16384

    return max_tokens


class LLMDiscordBot(commands.Bot, ABC):
    @abstractmethod
    def bot_token(self) -> str:
        """Discord bot token"""

    def prompt_task_dispatcher(self, user_id: str) -> TaskDispatcher:
        return TaskDispatcher([])

    def conversation_task_dispatcher(self, user_id: str) -> TaskDispatcher:
        return TaskDispatcher([])

    def monitored_channels(self) -> list[int]:
        return []

    def fully_readable_channels(self) -> list[int]:
        return self.monitored_channels()

    def conversation_llm_model(self, llm_convo_context: list[ChatMessage]) -> ChatOpenAI:
        used_tokens = sum([ChatOpenAI().get_num_tokens(m.content) for m in llm_convo_context])
        completion_tokens = 400

        required_tokens = used_tokens + completion_tokens
        if required_tokens < 2000:
            model_name = 'gpt-4'
        elif required_tokens < get_max_tokens('gpt-3.5-turbo'):
            model_name = 'gpt-3.5-turbo'
        else:
            model_name = 'gpt-3.5-turbo-16k'

        return ChatOpenAI(  # type: ignore
            model_name=model_name,
            temperature=0,
            max_tokens=completion_tokens,
        )

    def conversation_completion_tokens(self) -> int:
        return 400

    def conversation_system_prompt(self) -> str:
        return '''
You are ChatGPT, a large language model trained by OpenAI.
You are in a Discord chatroom.
Carefully heed the user's instructions.
        '''

    def is_authorized_to_read_message(self, message: Message) -> bool:
        if message.author == self.user:
            return True
        elif message.channel.id in self.fully_readable_channels():
            return True
        elif message.channel.type == ChannelType.private:
            return True
        else:
            return self.user in message.mentions

    def should_reply_to_message(self, message: Message) -> bool:
        if message.author == self.user:
            # Never reply to self, to avoid infinite loops
            return False
        elif message.channel.type == ChannelType.private:
            return True
        elif self.user in message.mentions:
            return True
        elif all([
            message.channel.id in self.monitored_channels(),
            not message.mentions,
            not message.author.bot,
        ]):
            return True
        else:
            return False

    async def on_message(self, message: Message) -> None:
        if not self.is_authorized_to_read_message(message):
            return

        if not self.should_reply_to_message(message):
            return

        async with message.channel.typing():
            reply = await self.reply(message)
        if reply:
            await message.channel.send(reply)

    async def reply(self, message: Message) -> Optional[str]:
        return await self.reply_to_message(message) or await self.reply_to_conversation(message)

    async def reply_to_message(self, message: Message) -> Optional[str]:
        return await self.prompt_task_dispatcher(str(message.author.id)).reply(
            self.remove_ai_mention(message.content),
            progress_reply_func=lambda reply: message.channel.send(reply),
        )

    async def reply_to_conversation(self, message: Message) -> Optional[str]:
        llm_convo_context = await self.get_llm_convo_context(latest_message=message)

        chat_model = self.conversation_llm_model(llm_convo_context)

        convo = ChatroomConversation(
            messages=llm_convo_context,
            ai_user_id=str(self.user.id),  # type: ignore
            max_tokens=get_max_tokens(chat_model.model_name) - self.conversation_completion_tokens()
        )

        return await self.conversation_task_dispatcher(str(message.author.id)).reply(
            convo.to_yaml(),
            progress_reply_func=lambda reply: message.channel.send(reply),
        )

    async def get_llm_convo_context(
        self,
        latest_message: Message,
    ) -> list[ChatroomMessage]:
        # Messages are ordered newest to oldest
        discord_messages = await self.preceding_messages(latest_message)

        chatroom_messages: List[ChatroomMessage] = []
        for msg in discord_messages:
            reply_reference = msg.reference.resolved if msg.reference and type(msg.reference.resolved) is Message else None

            chatroom_msg = ChatroomMessage(role=str(msg.author.id), content=msg.content, timestamp=int(msg.created_at.timestamp()))

            if reply_reference:
                chatroom_messages += [ChatMessage(role=str(reply_reference.author.id), content=reply_reference.content)]

            chatroom_messages += [chatroom_msg]

        return chatroom_messages

    async def preceding_messages(self, latest_message: Message) -> List[Message]:
        messages = [
            msg async for msg in latest_message.channel.history(
                after=arrow.now().shift(hours=-24).datetime,
                oldest_first=False,
                limit=20,
            )
            if self.is_authorized_to_read_message(msg)
        ]

        if latest_message.channel.type == ChannelType.public_thread:
            thread_start_msg = await latest_message.channel.parent.fetch_message(latest_message.channel.id)  # type: ignore
            messages += [thread_start_msg]

        return messages

    def remove_ai_mention(self, msg_content: str) -> str:
        return re.sub(rf'<@{self.user.id}> *', '', msg_content)  # type: ignore
