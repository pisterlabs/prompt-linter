"""Scaffolding to host your LangChain Chatbot on Steamship and connect it to Telegram."""
from typing import List, Optional, Type

from langchain.agents import Tool, initialize_agent, AgentType, AgentExecutor
from langchain.memory import ConversationBufferMemory
from pydantic import Field
from steamship.experimental.package_starters.telegram_bot import TelegramBot, TelegramBotConfig
from steamship.invocable import Config
from steamship_langchain.llms import OpenAIChat
from steamship_langchain.memory import ChatMessageHistory

from agent.base import LangChainAgentBot
from agent.prompts import PERSONALITY_PROMPT, SUFFIX, FORMAT_INSTRUCTIONS
from agent.tools.image import GenerateImageTool
from agent.tools.my_tool import MyTool
from agent.tools.reminder import RemindMe
from agent.tools.speech import GenerateSpeechTool
from agent.tools.video_message import VideoMessageTool

MODEL_NAME = "gpt-3.5-turbo"  # or "gpt-4.0"
TEMPERATURE = 0.7
VERBOSE = True


class ChatbotConfig(TelegramBotConfig):
    bot_token: str = Field(
        description="Your telegram bot token.\nLearn how to create one here: "
                    "https://github.com/steamship-packages/langchain-agent-production-starter/blob/main/docs/register-telegram-bot.md"
    )
    elevenlabs_api_key: str = Field(
        default="", description="Optional API KEY for ElevenLabs Voice Bot"
    )
    elevenlabs_voice_id: str = Field(
        default="", description="Optional voice_id for ElevenLabs Voice Bot"
    )
    chat_ids: str = Field(
        default="", description="Comma separated list of whitelisted chat_id's"
    )
    use_gpt4: bool = Field(
        True,
        description="If True, use GPT-4. Use GPT-3.5 if False. "
                    "GPT-4 generates better responses at higher cost and latency.",
    )


class LangChainTelegramChatbot(LangChainAgentBot, TelegramBot):
    """Deploy LangChain chatbots and connect them to Telegram."""

    config: ChatbotConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = "gpt-4" if self.config.use_gpt4 else "gpt-3.5-turbo"


    @classmethod
    def config_cls(cls) -> Type[Config]:
        """Return the Configuration class."""
        return ChatbotConfig

    def get_agent(self, chat_id: str) -> AgentExecutor:
        llm = OpenAIChat(
            client=self.client,
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            verbose=VERBOSE,
        )

        tools = self.get_tools(chat_id=chat_id)

        memory = self.get_memory(chat_id)

        return initialize_agent(
            tools,
            llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            agent_kwargs={
                "prefix": PERSONALITY_PROMPT,
                "suffix": SUFFIX,
                "format_instructions": FORMAT_INSTRUCTIONS,
            },
            verbose=VERBOSE,
            memory=memory,
        )

    def voice_tool(self) -> Optional[Tool]:
        """Return tool to generate spoken version of output text."""
        return GenerateSpeechTool(
            client=self.client,
            voice_id=self.config.elevenlabs_voice_id,
            elevenlabs_api_key=self.config.elevenlabs_api_key,
        )

    def get_memory(self, chat_id):
        if self.context and self.context.invocable_instance_handle:
            my_instance_handle = self.context.invocable_instance_handle
        else:
            my_instance_handle = "local-instance-handle"
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=ChatMessageHistory(
                client=self.client, key=f"history-{chat_id}-{my_instance_handle}"
            ),
            return_messages=True,
        )
        return memory

    def get_tools(self, chat_id: str) -> List[Tool]:
        return [
            # SearchTool(self.client),
            MyTool(self.client),
            GenerateImageTool(self.client),
            VideoMessageTool(self.client),
            RemindMe(invoke_later=self.invoke_later, chat_id=chat_id),
        ]
