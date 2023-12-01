from LunaBrainState import LunaBrainState, Intelligence
from chat.ChatMessage import ChatMessage
from chat.ChatResponseGenerator import ChatResponseGenerator
from chat.OpenAiChatGPT import OpenAiChatGPT
from chat.UsageTracker import UsageTracker


class LunaBrain(ChatResponseGenerator):

    def __init__(self,
                 open_ai_api_key: str,
                 usage_tracker: UsageTracker,
                 brain_state: LunaBrainState
                 ):
        self.gpt_4 = OpenAiChatGPT("gpt-4", open_ai_api_key, usage_tracker)
        self.gpt_3_5 = OpenAiChatGPT("gpt-3.5-turbo", open_ai_api_key, usage_tracker)
        self.brain_state = brain_state

    def __add_system_message(self, chat_messages: list[ChatMessage]):
        chat_messages.insert(0, ChatMessage(role="system", content=self.brain_state.system_message))

    def generate_chat_response(self, chat_messages: list[ChatMessage]) -> str:
        self.__add_system_message(chat_messages)
        match self.brain_state.intelligence:
            case Intelligence.Super:
                return self.gpt_4.generate_chat_response(chat_messages)
            case Intelligence.ChatGPT:
                return self.gpt_3_5.generate_chat_response(chat_messages)
            case _:
                return ""
