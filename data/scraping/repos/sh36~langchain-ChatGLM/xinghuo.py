import hmac
import hashlib
import base64
import json
import time
import urllib.parse
import websocket
import json
import requests

from abc import ABC
import requests
from typing import Optional, List
from langchain.llms.base import LLM

from models.loader import LoaderCheckPoint
from models.base import (RemoteRpcModel,
                         AnswerResult)
from typing import (
    Collection,
    Dict
)

class WsParam:
    def __init__(self, APPID, APIKey, APISecret, GPTURL):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.GPTURL = GPTURL
    
    def generate_signature(self):
        now = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())

        signature_origin = f"host: {self.get_host()}\n"
        signature_origin += f"date: {now}\n"
        signature_origin += f"GET {self.get_path()} HTTP/1.1"

        h = hmac.new(bytes(self.APISecret, "utf-8"), bytes(signature_origin, "utf-8"), hashlib.sha256)
        signature_sha = h.digest()
        signature_base64 = base64.b64encode(signature_sha).decode()

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_base64}"'
        authorization = base64.b64encode(authorization_origin.encode()).decode()

        return authorization
    
    def get_host(self):
        parsed_url = urllib.parse.urlparse(self.GPTURL)
        return parsed_url.netloc
    
    def get_path(self):
        parsed_url = urllib.parse.urlparse(self.GPTURL)
        return parsed_url.path
    
    def create_url(self):
        authorization = self.generate_signature()

        params = {
            "authorization": authorization,
            "date": time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime()),
            "host": self.get_host()
        }

        url = f"{self.GPTURL}?{urllib.parse.urlencode(params)}"
        return url

def gen_params(appID, question):
    data = {
        "header": {
            "app_id": appID,
            "uid": "1234"
        },
        "parameter": {
            "chat": {
                "domain": "general",
                "random_threshold": 0.5,
                "max_tokens": 2048,
                "auditing": "default"
            }
        },
        "payload": {
            "message": {
                "text": [
                    {"role": "user", "content": question}
                ]
            }
        }
    }

    params = json.dumps(data).encode()
    return params

def Xinghuo_chat(msg):
    appID = "89ed2ea4"
    apiKey = "0736d4e16d9cd2bb55d4b1bd38d07229"
    apiSecret = "MDlkZTRlYzNlZTcwZmU5YjU3ZmVmNWNj"
    gptURL = "wss://spark-api.xf-yun.com/v1.1/chat"
    question = msg
    reply = ""

    wsParam = WsParam(appID, apiKey, apiSecret, gptURL)
    url = wsParam.create_url()

    c = websocket.create_connection(url)
    
    try:
        data = gen_params(appID, question)

        c.send(data)

        while True:
            message = c.recv()
            response = json.loads(message)

            header = response["header"]
            code = header["code"]

            if code != 0:
                print(f"请求错误: {code}, {response}")
            else:
                payload = response["payload"]
                choices = payload["choices"]
                status = choices["status"]
                content = choices["text"][0]["content"]
                reply += content

                if status == 2:
                    break
    
    finally:
        c.close()

    print("星火回复：" + reply)
    return reply

def Xinghuo_conversation(sender, msg):
    # 读取存储的历史记录
    # TODO 根据wx_id获取历史对话

    reply = Xinghuo_chat(msg)
    return reply

def _build_message_template() -> Dict[str, str]:
    """
    :return: 结构
    """
    return {
        "role": "",
        "content": "",
    }


class XINGHUOLLM(RemoteRpcModel, LLM, ABC):
    api_base_url: str = "wss://spark-api.xf-yun.com/v1.1/chat"
    model_name: str = "xinghuo"
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    checkPoint: LoaderCheckPoint = None
    history = []
    history_len: int = 10

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "xinghuo"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    @property
    def _api_key(self) -> str:
        pass

    @property
    def _api_base_url(self) -> str:
        return self.api_base_url

    def set_api_key(self, api_key: str):
        pass

    def set_api_base_url(self, api_base_url: str):
        self.api_base_url = api_base_url

    def call_model_name(self, model_name):
        self.model_name = model_name

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        pass

    # 将历史对话数组转换为文本格式
    def build_message_list(self, query) -> Collection[Dict[str, str]]:
        build_message_list: Collection[Dict[str, str]] = []
        history = self.history[-self.history_len:] if self.history_len > 0 else []
        for i, (old_query, response) in enumerate(history):
            user_build_message = _build_message_template()
            user_build_message['role'] = 'user'
            user_build_message['content'] = old_query
            system_build_message = _build_message_template()
            system_build_message['role'] = 'system'
            system_build_message['content'] = response
            build_message_list.append(user_build_message)
            build_message_list.append(system_build_message)

        user_build_message = _build_message_template()
        user_build_message['role'] = 'user'
        user_build_message['content'] = query
        build_message_list.append(user_build_message)
        return build_message_list

    def generatorAnswer(self, prompt: str,
                        history: List[List[str]] = [],
                        streaming: bool = False):

        # create a chat completion

        reply = Xinghuo_chat(prompt)
        # completion = openai.ChatCompletion.create(
        #     model=self.model_name,
        #     messages=self.build_message_list(prompt)
        # )

        history += [[prompt, reply]]
        answer_result = AnswerResult()
        answer_result.history = history
        answer_result.llm_output = {"answer": reply}

        yield answer_result
