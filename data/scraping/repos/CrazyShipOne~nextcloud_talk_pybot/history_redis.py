from ncbot.plugins.utils.history import MemoryHistoryUtil
from langchain.memory import ChatMessageHistory
import redis
import json
import ncbot.config as ncconfig

conn = redis.Redis(host=ncconfig.cf.redis_host, port=ncconfig.cf.redis_port, db=ncconfig.cf.redis_db,password=ncconfig.cf.redis_pass)


class RedisMemoryHistoryUtil(MemoryHistoryUtil):

    def __init__(self) -> None:
        super().__init__()

    def _save_to_memory(self, userid, history):
        index_key = super()._get_index_key(userid)
        push_list = history[-2:]
        for ele in push_list:
            conn.rpush(index_key, json.dumps(ele))

    def _get_from_memory(self, userid):
        if not super()._isStore():
            return None
        index_key = super()._get_index_key(userid)
        dict_range = conn.lrange(index_key,0, -1)
        dict = [json.loads(m.decode('utf-8')) for m in dict_range]
        return dict
    

    def clear_memory(self, userid):
        conn.delete(super()._get_index_key(userid))