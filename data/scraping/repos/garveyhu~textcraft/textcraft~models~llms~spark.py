import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
import logging
import ssl
from datetime import datetime
from time import mktime
from typing import Any, List, Mapping, Optional
from urllib.parse import urlencode, urlparse
from wsgiref.handlers import format_date_time

import websocket
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.llms.base import LLM

from textcraft.core.config import default_model, keys_spark, model_temperature

logging.basicConfig(level=logging.INFO)
set_llm_cache(InMemoryCache())
result_list = []


def _construct_query(prompt, temperature, max_tokens, appid, domain):
    data = {
        "header": {"app_id": appid, "uid": "1234"},
        "parameter": {
            "chat": {
                "domain": domain,
                "random_threshold": temperature,
                "max_tokens": max_tokens,
                "auditing": "default",
            }
        },
        "payload": {"message": {"text": [{"role": "user", "content": prompt}]}},
    }
    return data


def _run(ws, *args):
    data = json.dumps(
        _construct_query(
            prompt=ws.question,
            temperature=ws.temperature,
            max_tokens=ws.max_tokens,
            appid=ws.appid,
            domain=ws.domain,
        )
    )
    print(data)
    ws.send(data)


def on_error(ws, error):
    print("error:", error)


def on_close(ws):
    print("closed...")


def on_open(ws):
    thread.start_new_thread(_run, (ws,))


def on_message(ws, message):
    data = json.loads(message)
    print("====data=====")
    print(data)
    code = data["header"]["code"]
    # print(data)
    if code != 0:
        print(f"请求错误: {code}, {data}")
        ws.close()
    else:
        choices = data["payload"]["choices"]
        status = choices["status"]
        content = choices["text"][0]["content"]
        result_list.append(content)
        if status == 2:
            ws.close()
            setattr(ws, "content", "".join(result_list))
            print("====result_list=====")
            print(result_list)
            result_list.clear()


class Spark(LLM):
    """
    根据源码解析在通过LLMS包装的时候主要重构两个部分的代码
    _call 模型调用主要逻辑,输入问题，输出模型相应结果
    _identifying_params 返回模型描述信息，通常返回一个字典，字典中包括模型的主要参数
    """

    modelDict = {
        "spark-v2": "ws://spark-api.xf-yun.com/v2.1/chat",
        "spark-v3": "ws://spark-api.xf-yun.com/v3.1/chat",
    }
    domainDict = {"spark-v2": "generalv2", "spark-v3": "generalv3"}
    max_tokens = 1024

    @property
    def _llm_type(self) -> str:
        return "Spark"

    def _get_url(self, api_key, api_secret):
        modelName = default_model()
        gpt_url = self.modelDict.get(modelName)
        host = urlparse(gpt_url).netloc
        path = urlparse(gpt_url).path

        print(gpt_url)

        """
        Generates a URL with authorization headers for making a GET request to the Spark API.

        Returns:
        str: The URL with authorization headers.
        """
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        signature_origin = "host: " + host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + path + " HTTP/1.1"

        signature_sha = hmac.new(
            api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding="utf-8")

        authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(
            encoding="utf-8"
        )

        v = {"authorization": authorization, "date": date, "host": host}
        url = gpt_url + "?" + urlencode(v)
        return url

    def _post(self, prompt):
        websocket.enableTrace(False)
        SPARK_APPID, SPARK_API_KEY, SPARK_API_SECRET = keys_spark()
        wsUrl = self._get_url(SPARK_API_KEY, SPARK_API_SECRET)
        ws = websocket.WebSocketApp(
            wsUrl,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open,
        )
        ws.question = prompt
        temperature = model_temperature()
        domain = self.domainDict.get(default_model())
        setattr(ws, "temperature", temperature)
        setattr(ws, "appid", SPARK_APPID)
        setattr(ws, "max_tokens", self.max_tokens)
        setattr(ws, "domain", domain)
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        return ws.content if hasattr(ws, "content") else ""

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        content = self._post(prompt)
        print("======content======")
        print(content)
        return content

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Get the identifying parameters.
        """
        modelName = default_model()
        gpt_url = self.modelDict.get(modelName)
        _param_dict = {"url": gpt_url}
        return _param_dict
