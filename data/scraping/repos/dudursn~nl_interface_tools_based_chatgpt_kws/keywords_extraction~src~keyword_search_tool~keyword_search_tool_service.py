
from .keyword_search_tool_config import KeywordSearchToolConfig
from openai_model.chatgpt import run_conversation
from interface.service_interface import ServiceInterface

class KeywordSearchToolService(ServiceInterface):
    def __init__(self):
        self.__kws_config = KeywordSearchToolConfig()

    def execute(self, user_input, opt="1"):
        return run_conversation(user_input, self.__kws_config.get_config_function(), opt)
 