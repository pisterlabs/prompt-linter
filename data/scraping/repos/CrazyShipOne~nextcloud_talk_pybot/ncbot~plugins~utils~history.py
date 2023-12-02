
from abc import abstractmethod
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
import ncbot.config as ncconfig

class MemoryHistoryUtil():
    

    def __init__(self):
        self.max_chat_history = ncconfig.cf.max_chat_history
        self.save_type = ncconfig.cf.save_type
    

    def _isStore(self):
        return self.max_chat_history != 0


    @abstractmethod
    def _save_to_memory(self, userid, history):
        pass

    @abstractmethod
    def _get_from_memory(self, userid):
        pass


    @abstractmethod
    def clear_memory(self, userid):
        pass


    def get_memory(self, userid):
        dict = self._get_from_memory(userid)
        if dict == None or len(dict) == 0:
            return ConversationBufferMemory()
        memory_dict = self.__dict_to_message(dict)
        history = ChatMessageHistory()
        history.messages = history.messages + memory_dict
        return ConversationBufferMemory(chat_memory=history)


    def save_memory(self, userid, history: ConversationBufferMemory):
        chat_memory = history.chat_memory
        memory = self.__tuncate_memory(chat_memory)
        self._save_to_memory(userid, memory)


    def __tuncate_memory(self, history):
        memory_dict = self.__message_to_dict(history)
        if len(memory_dict) > self.max_chat_history * 2:
            memory_dict = memory_dict[2:]
        return memory_dict


    def _get_index_key(self, userid):
        return f'memory_{userid}'
    

    def __message_to_dict(self, history: ChatMessageHistory):
        return messages_to_dict(history.messages)
    

    def __dict_to_message(self, load_dict):
        return messages_from_dict(load_dict)
    

from ncbot.plugins.utils.history_memory import InMemoryHistoryUtil
from ncbot.plugins.utils.history_redis import RedisMemoryHistoryUtil

in_memory_util = InMemoryHistoryUtil()
redis_memory_util = RedisMemoryHistoryUtil()

def get_instance():
    match ncconfig.cf.save_type:
        case 'memory':
            return in_memory_util
        case 'redis':
            return redis_memory_util