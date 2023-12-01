from json import dumps as jsonDumps
from json import loads as jsonLoads
from pathlib import Path
from typing import Union, List
import openai


try:
    import aiohttp
    from openai import api_requestor
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def aiohttp_session():
        """
        该函数是基于 PyPi包 "openai" 中的 aiohttp_session 函数改写
        """
        user_set_session = openai.aiosession.get()
        if user_set_session:
            yield user_set_session
        else:
            async with aiohttp.ClientSession(trust_env=True) as session:
                yield session

    api_requestor.aiohttp_session = aiohttp_session
except:
    pass


class AKPool:
    """轮询获取api_key"""

    def __init__(self, apikeys: list):
        self._pool = self._POOL(apikeys)

    def fetch_key(self):
        return next(self._pool)

    @classmethod
    def _POOL(cls, apikeys: list):
        while True:
            for x in apikeys:
                yield x


class RoleMsgBase:
    role_name: str
    text: str

    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

    def __iter__(self):
        yield "role", self.role_name
        yield "content", self.text


system_msg = type("system_msg", (RoleMsgBase,), {"role_name": "system"})
user_msg = type("user_msg", (RoleMsgBase,), {"role_name": "user"})
assistant_msg = type("assistant_msg", (RoleMsgBase,), {"role_name": "assistant"})


class Temque:
    """一个先进先出, 可设置最大容量, 可固定元素的队列"""

    def __init__(self, maxlen: int = None):
        self.core: List[dict] = []
        self.maxlen = maxlen or float("inf")

    def _trim(self):
        core = self.core
        if len(core) > self.maxlen:
            dc = len(core) - self.maxlen
            indexes = []
            for i, x in enumerate(core):
                if not x["pin"]:
                    indexes.append(i)
                if len(indexes) == dc:
                    break
            for i in indexes[::-1]:
                core.pop(i)

    def add_many(self, *objs):
        for x in objs:
            self.core.append({"obj": x, "pin": False})
        self._trim()

    def __iter__(self):
        for x in self.core:
            yield x["obj"]

    def pin(self, *indexes):
        for i in indexes:
            self.core[i]["pin"] = True

    def unpin(self, *indexes):
        for i in indexes:
            self.core[i]["pin"] = False

    def copy(self):
        que = self.__class__(maxlen=self.maxlen)
        que.core = self.core.copy()
        return que

    def deepcopy(self):
        ...  # 创建这个方法是为了提醒用户: copy 方法是浅拷贝

    def __add__(self, obj: Union[list, "Temque"]):
        que = self.copy()
        if isinstance(obj, self.__class__):
            que.core += obj.core
            que._trim()
        else:
            que.add_many(*obj)
        return que


class Chat:
    """
    文档: https://pypi.org/project/openai2

    获取api_key:
        获取链接1: https://platform.openai.com/account/api-keys
        获取链接2: https://www.baidu.com/s?wd=%E8%8E%B7%E5%8F%96%20openai%20api_key
    """

    recently_used_apikey: str = ""

    def __init__(
        self,
        api_key: Union[str, AKPool],
        model: str = "gpt-3.5-turbo",
        MsgMaxCount=None,
        **kwargs,
    ):
        self.reset_api_key(api_key)
        self.model = model
        self._messages = Temque(maxlen=MsgMaxCount)
        self.kwargs = kwargs

    def reset_api_key(self, api_key: Union[str, AKPool]):
        if isinstance(api_key, AKPool):
            self._akpool = api_key
        else:
            self._akpool = AKPool([api_key])

    def request(self, text: str):
        self.recently_used_apikey = self._akpool.fetch_key()
        completion = openai.ChatCompletion.create(
            **{
                "api_key": self.recently_used_apikey,
                "model": self.model,
                "messages": list(self._messages + [{"role": "user", "content": text}]),
                **self.kwargs,
            }
        )
        answer: str = completion.choices[0].message["content"]
        self._messages.add_many(
            {"role": "user", "content": text}, {"role": "assistant", "content": answer}
        )
        return answer

    def stream_request(self, text: str):
        self.recently_used_apikey = self._akpool.fetch_key()
        completion = openai.ChatCompletion.create(
            **{
                "api_key": self.recently_used_apikey,
                "model": self.model,
                "messages": list(self._messages + [{"role": "user", "content": text}]),
                "stream": True,
                **self.kwargs,
            }
        )
        answer: str = ""
        for chunk in completion:
            choice = chunk.choices[0]
            if choice.finish_reason == "stop":
                break
            content: str = choice.delta.get("content", "")
            answer += content
            yield content
        self._messages.add_many(
            {"role": "user", "content": text}, {"role": "assistant", "content": answer}
        )

    async def asy_request(self, text: str):
        self.recently_used_apikey = self._akpool.fetch_key()
        completion = await openai.ChatCompletion.acreate(
            **{
                "api_key": self.recently_used_apikey,
                "model": self.model,
                "messages": list(self._messages + [{"role": "user", "content": text}]),
                **self.kwargs,
            }
        )
        answer: str = completion.choices[0].message["content"]
        self._messages.add_many(
            {"role": "user", "content": text}, {"role": "assistant", "content": answer}
        )
        return answer

    async def async_stream_request(self, text: str):
        self.recently_used_apikey = self._akpool.fetch_key()
        completion = await openai.ChatCompletion.acreate(
            **{
                "api_key": self.recently_used_apikey,
                "model": self.model,
                "messages": list(self._messages + [{"role": "user", "content": text}]),
                "stream": True,
                **self.kwargs,
            }
        )
        answer: str = ""
        async for chunk in completion:
            choice = chunk.choices[0]
            if choice.finish_reason == "stop":
                break
            content: str = choice.delta.get("content", "")
            answer += content
            yield content
        self._messages.add_many(
            {"role": "user", "content": text}, {"role": "assistant", "content": answer}
        )

    def rollback(self, n=1):
        self._messages.core[-2 * n :] = []
        for x in self._messages.core[-2:]:
            x = x["obj"]
            print(f"[{x['role']}]:{x['content']}")

    def pin(self, *indexes):
        self._messages.pin(*indexes)

    def unpin(self, *indexes):
        self._messages.unpin(*indexes)

    def dump(self, fpath: str):
        """存档"""
        jt = jsonDumps(list(self._messages), ensure_ascii=False)
        Path(fpath).write_text(jt, encoding="utf8")
        return True

    def load(self, fpath: str):
        """载入存档"""
        jt = Path(fpath).read_text(encoding="utf8")
        self._messages.add_many(*jsonLoads(jt))
        return True

    def forge(self, *messages: Union[system_msg, user_msg, assistant_msg]):
        """伪造对话内容"""
        for x in messages:
            self._messages.add_many(dict(x))
        print(self._messages)

    def fetch_messages(self):
        return list(self._messages)
