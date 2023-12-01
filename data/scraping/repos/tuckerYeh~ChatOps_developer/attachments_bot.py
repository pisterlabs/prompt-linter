import openai
import tiktoken
from botbuilder.core import ActivityHandler, MessageFactory, TurnContext, CardFactory
from botbuilder.schema import (
    ChannelAccount,
    HeroCard,
    CardAction,
    ActivityTypes,
    Attachment,
    AttachmentData,
    Activity,
    ActionTypes,
)
from botbuilder.core.memory_storage import MemoryStorage
from botbuilder.core.conversation_state import ConversationState
import sys
sys.path.append("..")
from config import DefaultConfig
#from langchain_google import GoogleSearchWeb

CONFIG = DefaultConfig()


class ConversationData:
    def __init__(self, conversation=None):
        self.conversation = conversation or []

class AttachmentsBot(ActivityHandler):
    def __init__(self):
        memory_storage = MemoryStorage()
        self.conversation_state = ConversationState(memory_storage)
        self.conversation_data_accessor = self.conversation_state.create_property("ConversationData")


    async def on_members_added_activity(self, members_added: [ChannelAccount], turn_context: TurnContext):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("您好！我是TuckerAI,提供了ChatGPT的功能,只需輸入您的問題或要求,就會立即為您提供答案或解決方案!")

    async def on_message_activity(self, turn_context: TurnContext):

        openai.api_type = "azure"
        openai.api_version = "2023-05-15"
        openai.api_base = CONFIG.APP_AZURE_OPENAIAPIBASE
        openai.api_key = CONFIG.APP_AZURE_OPENAIAPIKEY
        system_message = {"role": "system", "content": "使用繁體中文，.\
                                        簡潔答覆，忽略禮貌用語"}
        max_response_tokens = 1024
        token_limit = 4096

        # 從對話狀態中取得對話
        conversation_data = await self.conversation_data_accessor.get(turn_context, ConversationData)
        conversation = conversation_data.conversation

        user_input = turn_context.activity.text
        conversation.append(system_message)
        conversation.append({"role": "user", "content": user_input})

        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = sum(len(encoding.encode(msg["content"])) + 4 for msg in conversation) + 2

        while num_tokens + max_response_tokens >= token_limit:
            del conversation[1]
            num_tokens = sum(len(encoding.encode(msg["content"])) + 4 for msg in conversation) + 2

        response = openai.ChatCompletion.create(
            engine="kgis-gpt35",
            #engine="tuckerai-gpt4",
            messages=conversation,
            temperature=0.7,
            max_tokens=max_response_tokens,
        )

        conversation.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
        reply_msg = response['choices'][0]['message']['content']

        # 更新對話
        conversation_data.conversation = conversation
        await self.conversation_state.save_changes(turn_context)

        await turn_context.send_activity(reply_msg)

    #錯誤訊息
    async def _send_unrecognizable_message(self, turn_context: TurnContext):
        await turn_context.send_activity(f"TuckerAI罷工中．請重新輸入")
        await self._display_index(turn_context)



