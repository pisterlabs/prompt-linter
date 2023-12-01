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

ErnieBotURL = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant"


class ErnieBotResponseBody:
    def __init__(self):
        self.ID = ""
        self.Object = ""
        self.Created = 0
        self.SentenceID = 0
        self.IsEnd = False
        self.Result = ""
        self.NeedClearHistory = False


class ErnieBotMessage:
    def __init__(self, role="", content=""):
        self.role = role
        self.content = content
    
    def to_dict(self):
        return {
            'role': self.role,
            'content': self.content
        }


class ErnieBotRequestBody:
    def __init__(self, messages=None, stream=False, user_id=""):
        if messages is None:
            messages = []
        self.messages = list(map(lambda msg: msg.to_dict(), messages))
        self.stream = stream
        self.user_id = user_id
        self.__dict__ = self.to_dict()

    def to_dict(self):
        return {
            'messages': self.messages,
            'stream': self.stream,
            'user_id': self.user_id
        }


class AccessTokenResponse:
    def __init__(self):
        self.AccessToken = ""


class ErnieBotChatHistory:
    def __init__(self):
        self.History = []
        self.Conversations = 0


chatHistory = ErnieBotChatHistory()


def get_access_token(client_id, client_secret):
    url = "https://aip.baidubce.com/oauth/2.0/token"
    payload = f"grant_type=client_credentials&client_id={client_id}&client_secret={client_secret}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, data=payload, headers=headers)
    response_dict = json.loads(response.text)
    access_token_response = AccessTokenResponse()
    access_token_response.AccessToken = response_dict.get('access_token', '')
    return access_token_response.AccessToken


def ernie_bot_chat(msg):
    access_token = get_access_token("aX9xYA7eQi9nvMF2cRwyDG0q", "NrWvvEPBIeqLRwridSr3RUqtd5CZhUcA")
    request_body = ErnieBotRequestBody(
        messages=[
            ErnieBotMessage(role="user", content=msg)
        ],
        stream=False,
        user_id=""
    )
    request_data = json.dumps(request_body.to_dict())
    new_ernie_bot_url = f"{ErnieBotURL}?access_token={access_token}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.post(new_ernie_bot_url, data=request_data, headers=headers)
    response_dict = json.loads(response.text)
    ernie_bot_response_body = ErnieBotResponseBody()
    ernie_bot_response_body.ID = response_dict.get('id', '')
    ernie_bot_response_body.Object = response_dict.get('object', '')
    ernie_bot_response_body.Created = response_dict.get('created', 0)
    ernie_bot_response_body.SentenceID = response_dict.get('sentence_id', 0)
    ernie_bot_response_body.IsEnd = response_dict.get('is_end', False)
    ernie_bot_response_body.Result = response_dict.get('result', '')
    ernie_bot_response_body.NeedClearHistory = response_dict.get('need_clear_history', False)
    reply = ernie_bot_response_body.Result
    print("文心回复：" + reply)
    if reply == '':
        reply = '很抱歉，此问题无法回答，请稍后再问。'
    return reply


def ernie_bot_conversation(sender, msg):
    reply = ernie_bot_chat(msg)
    return reply


def chat_history_clear():
    chatHistory.History = []
    chatHistory.Conversations = 0

def _build_message_template() -> Dict[str, str]:
    """
    :return: 结构
    """
    return {
        "role": "",
        "content": "",
    }


class ERNIELLM(RemoteRpcModel, LLM, ABC):
    api_base_url: str = ErnieBotURL
    model_name: str = "ernie"
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
        return "ERNIE"

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

        reply = ernie_bot_chat(prompt)
        # completion = openai.ChatCompletion.create(
        #     model=self.model_name,
        #     messages=self.build_message_list(prompt)
        # )

        history += [[prompt, reply]]
        answer_result = AnswerResult()
        answer_result.history = history
        answer_result.llm_output = {"answer": reply}

        yield answer_result
