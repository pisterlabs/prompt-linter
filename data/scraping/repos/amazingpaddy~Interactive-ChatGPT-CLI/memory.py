from langchain.callbacks.base import CallbackManager
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

from chat_gpt import ChatGPT
from config import OPENAI_API_KEY, CHAT_SETTINGS
from streaming_handler import CustomStreamingStdOutCallbackHandler


class MemoryManager:
    """
    MemoryManager is a class that manages memory-related functionalities for the chat application.
    It initializes and manages ConversationSummaryBufferMemory and ConversationChain objects.
    """

    def __init__(self, chat_gpt: ChatGPT, is_stream: bool = False):
        """
        Initializes a new MemoryManager instance.

        :param chat_gpt: ChatGPT object to be used for language generation.
        :param is_stream: If True, use streaming mode (currently not implemented).
        """
        self.llm = None
        self.conversation_chain = None
        self.chat_gpt = chat_gpt
        self.init_memory(is_stream)

    def init_memory(self, is_stream: bool):
        """
        Initializes memory-related components.

        :param is_stream: If True, use streaming mode.
        """
        if is_stream:
            callback_manager = CallbackManager([CustomStreamingStdOutCallbackHandler(self.chat_gpt.console)])
            self.llm = ChatOpenAI(temperature=CHAT_SETTINGS['temperature'],
                                  openai_api_key=OPENAI_API_KEY,
                                  model_name=self.chat_gpt.model_name,
                                  max_tokens=CHAT_SETTINGS['max_tokens'],
                                  streaming=True,
                                  callback_manager=callback_manager,
                                  verbose=True)
        else:
            # Initialize the ChatOpenAI instance for language generation.
            self.llm = ChatOpenAI(temperature=CHAT_SETTINGS['temperature'],
                                  openai_api_key=OPENAI_API_KEY,
                                  model_name=self.chat_gpt.model_name,
                                  max_tokens=CHAT_SETTINGS['max_tokens'],
                                  verbose=True)

        # Initialize the ConversationChain instance for managing the conversation process.
        self.conversation_chain = ConversationChain(llm=self.llm,
                                                    memory=ConversationSummaryBufferMemory(llm=self.llm,
                                                                                           max_token_limit=1500,
                                                                                           return_messages=True))

    def reset_memory(self, is_stream: bool):
        self.init_memory(is_stream)
