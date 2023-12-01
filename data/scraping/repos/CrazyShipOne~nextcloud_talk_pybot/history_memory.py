from ncbot.plugins.utils.history import MemoryHistoryUtil
from langchain.memory import ChatMessageHistory


class InMemoryHistoryUtil(MemoryHistoryUtil):

    def __init__(self) -> None:
        super().__init__()
        self.__memories = {}

    def _save_to_memory(self, userid, history):
        index_key = super()._get_index_key(userid)
        self.__memories[index_key] = history

    def _get_from_memory(self, userid):
        if not super()._isStore():
            return None
        index_key = super()._get_index_key(userid)
        if not index_key in self.__memories:
            return None
        return self.__memories[index_key]
    

    def clear_memory(self, userid):
        del self.__memories[super()._get_index_key(userid)]