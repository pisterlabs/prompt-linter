import asyncio
import html
import os
import re

from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from maubot import MessageEvent, Plugin
from maubot.config import Config as maubot_config
from maubot.handlers import command, event
from mautrix.types import EventType, Membership, StateEvent
from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper


class Config(BaseProxyConfig):
    def do_update(self, helper: ConfigUpdateHelper) -> None:
        helper.copy("OPENAI_API_KEY")
        helper.copy("SERPAPI_API_KEY")


class AIBot(Plugin):
    @classmethod
    def get_config_class(cls) -> type[BaseProxyConfig]:
        return Config

    async def start(self) -> None:
        self.config.load_and_update()

    MAX_INPUT_LENGTH = 1000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        os.environ["LANGCHAIN_HANDLER"] = "langchain"
        os.environ["SERPAPI_API_KEY"] = self.config["SERPAPI_API_KEY"]

        config = maubot_config("config.yaml", "standalone/example-config.yaml")
        config.load()

        self.user_id = config["user.credentials.id"]
        self.server = config["user.credentials.homeserver"]
        self.token = config["user.credentials.access_token"]

        self.conversations = {}
        self.gpt_versions = {}
        self.join_message_lock = asyncio.Lock()

    @event.on(EventType.ROOM_MEMBER)
    async def handle_join(self, evt: StateEvent) -> None:
        if evt.content.membership != Membership.JOIN:
            return

        if evt.state_key != self.client.mxid:
            return

        async with self.join_message_lock:
            # Check if the join message has been sent using room account data
            try:
                join_message_sent_data = await self.client.get_account_data(
                    "maubot.join_message_sent", evt.room_id
                )
            except:
                join_message_sent_data = None
            if join_message_sent_data and join_message_sent_data.get(
                "join_message_sent"
            ):
                return

            # Set the room account data to indicate that the join message has been sent
            await self.client.set_account_data(
                "maubot.join_message_sent", {"join_message_sent": True}, evt.room_id
            )

            room_members = await self.get_joined_members(evt.room_id)
            num_members = len(room_members)

            # Fetch the bot's display name
            display_name = evt.content.displayname

            if len(room_members) == 2 or (
                len(room_members) == 3
                and any(member.endswith("bot:beeper.local") for member in room_members)
            ):
                message = f"I am your friendly neighbourhood Beeper AI! <br><br>I am powered by Beeper.com and ChatGPT. All messages in this chat will be shared with Beeper and OpenAI."
            else:
                message = f"I am your friendly neighbourhood Beeper AI! To ask me something, send a message starting with @AI or mention #AI.<br><br> I am powered by Beeper.com and ChatGPT. All messages in this chat will be shared with Beeper and OpenAI."

            await self.client.send_message_event(
                evt.room_id,
                EventType.ROOM_MESSAGE,
                {
                    "msgtype": "m.text",
                    "body": html.unescape(
                        re.sub("<[^<]+?>", "", message)
                    ),  # Plain text version of the message
                    "format": "org.matrix.custom.html",
                    "formatted_body": message,
                },
            )

    @command.passive(regex=r"^[^!].*")
    async def process_message(self, event: MessageEvent, _: str) -> None:
        # Check if the room has only 2 members or 3 members with one bot
        room_members = await self.get_joined_members(event.room_id)
        should_reply = False
        if len(room_members) == 2 or (
            len(room_members) == 3
            and any(member.endswith("bot:beeper.local") for member in room_members)
        ):
            should_reply = True
        else:
            mention_data = await self.is_bot_mentioned(event)
            if mention_data:
                should_reply = True

        if should_reply:
            input_text = event.content["body"][: self.MAX_INPUT_LENGTH]
            response_text = await self.chat(input_text, event.room_id)
            await event.reply(response_text)

    @command.new(name="gpt4")
    async def switch_to_gpt4(self, event: MessageEvent) -> None:
        if self.gpt_versions.get(event.room_id) == "gpt-4":
            await event.reply("This room is already using GPT-4.")
        else:
            self.gpt_versions[event.room_id] = "gpt-4"
            await event.reply(
                "This room has been switched to GPT-4. If you would like to undo this, type !gpt3.5"
            )

    @command.new(name="gpt3.5")
    async def switch_to_gpt3_5(self, event: MessageEvent) -> None:
        # .get allows it to work even if event.room_id isn't defined (if the user types !gpt3.5 without sending a message beforehand)
        # the second part of the statement checks whether room_id is defined, since if not it initializes as gpt-3.5-turbo
        if (
            self.gpt_versions.get(event.room_id) == "gpt-3.5-turbo"
            or not event.room_id in self.gpt_versions
        ):
            await event.reply("This room is already using GPT-3.5.")
        else:
            self.gpt_versions[event.room_id] = "gpt-3.5-turbo"
            await event.reply(
                "This room has been switched to GPT-3.5. If you would like to undo this, type !gpt4"
            )

    async def get_joined_members(self, room_id):
        room_members = []
        response = await self.client.get_joined_members(room_id)
        for user_id, member_info in response.items():
            room_members.append(user_id)
        return room_members

    async def is_bot_mentioned(self, event: MessageEvent):
        message_text = event.content.get("body", "")
        # print(event)
        if not message_text:
            return False

        # TODO: @griffinai vs @ai, if the bot isn't named "ai"
        mention_pattern = re.compile(r"^(AI|ai|@AI|@ai)|[#]AI|[#]ai")
        match = mention_pattern.search(message_text)

        if match:
            start, end = match.start(), match.end()
            text = message_text[end:].strip()
            return start, end, text

        relates_to = event.content.get("_relates_to", "")
        if relates_to:
            in_reply_to = relates_to.get("in_reply_to", "")
            if in_reply_to:
                in_reply_to_event_id = in_reply_to.get("event_id", "")
                if in_reply_to_event_id:
                    in_reply_to_event = await self.client.get_event(
                        event.room_id, in_reply_to_event_id
                    )

                    if in_reply_to_event.sender == self.client.mxid:
                        text = message_text.strip()
                        return 0, len(text), text

        return None

    async def typing(self, status: bool, time: int, room_id: str):
        await self.http.put(
            f"{self.server}/_matrix/client/v3/rooms/{room_id}/typing/{self.user_id}",
            json={"typing": status, "timeout": time},
            headers={"Authorization": f"Bearer {self.token}"},
        )

    async def chat(self, text: str, room_id: str) -> str:
        try:
            if len(text) > self.MAX_INPUT_LENGTH:
                return f"Input text exceeds maximum length of {self.MAX_INPUT_LENGTH} characters."

            if room_id not in self.gpt_versions:
                self.gpt_versions[room_id] = "gpt-3.5-turbo"

            # add typing indicator
            await self.typing(True, 60000, room_id)

            llm = ChatOpenAI(
                temperature=0,
                model_name=self.gpt_versions[room_id],
                openai_api_key=self.config["OPENAI_API_KEY"],
            )
            tools = load_tools(["serpapi", "llm-math", "wikipedia"], llm=llm)

            if room_id not in self.conversations:
                self.conversations[room_id] = ConversationSummaryBufferMemory(
                    llm=llm,
                    max_token_limit=200,
                    memory_key="chat_history",
                    return_messages=True,
                )

            memory = self.conversations[room_id]

            # removed: verbose=True,
            agent_chain = initialize_agent(
                tools,
                llm,
                agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
            )

            response = agent_chain.run(text.strip())

            # remove typing indicator
            await self.typing(False, 0, room_id)

            return response

        except Exception as e:
            # remove typing indicator
            await self.typing(False, 0, room_id)

            self.log.error(f"Error in GPT API call: {e}")
            return "An error occurred while processing your request. Please check the logs for more information."
