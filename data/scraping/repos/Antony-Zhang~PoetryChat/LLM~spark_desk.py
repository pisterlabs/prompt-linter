"""
    使用LangChain自定义LLM
"""
import os
from dotenv import find_dotenv, load_dotenv


from typing import Optional, List, Mapping, Any
import ssl

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

import websocket
from LLM.webInteract.web_param import WsParamGPT
from LLM.webInteract.web_interact_gpt import Singleton, WS
from LLM.webInteract.web_interact_gpt import (on_close,
                                              on_open,
                                              on_error,
                                              on_message)


@Singleton
class SparkDesk(LLM):
    """
    讯飞星火的语言模型
    """

    url = "wss://spark-api.xf-yun.com/v1.1/chat"

    # 本地调试时使用，加载.env文件
    load_dotenv(find_dotenv('.env'))
    APPID = os.getenv("APPID_LLM")      # 环境变量
    APIKey = os.getenv("APIKEY_LLM")
    APISecret = os.getenv("APISECRET_LLM")

    # # HF部署时使用，直接加载Hub上设置的环境变量
    # APPID = os.environ.get("APPID")
    # APIKey = os.getenv("APIKEY")
    # APISecret = os.getenv("APISECRET")

    @property
    def _llm_type(self) -> str:
        return "SparkDesk"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        _param_dict = {
            "url": self.url,
            "APPID": self.APPID,
            "APIKey": self.APIKey,
            "APISecret": self.APISecret
        }
        return _param_dict

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        ws_param = WsParamGPT(self.url, self.APPID, self.APIKey, self.APISecret)
        websocket.enableTrace(False)
        wsUrl = ws_param.create_url()
        ws = WS(appid=ws_param.APPID,
                url=wsUrl,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open)
        ws.question = prompt
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})  # 建立长连接
        return ws.received_message


if __name__ == "__main__":
    llm = SparkDesk()

    # test
    prompt_1 = PromptTemplate(
        input_variables=["lastname"],
        template="我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字",
    )
    chain_1 = LLMChain(llm=llm,
                       prompt=prompt_1)
    # 创建第二条链
    prompt_2 = PromptTemplate(
        input_variables=["child_name"],
        template="邻居的儿子名字叫{child_name}，给他起一个小名",
    )
    chain_2 = LLMChain(llm=llm, prompt=prompt_2)

    # 链接两条链
    overall_chain = SimpleSequentialChain(chains=[chain_1, chain_2], verbose=True)

    # 执行链，只需要传入第一个参数
    catchphrase = overall_chain.run("王")
    print(catchphrase)

# # 使用代理来确定如何使用LLM来采取行动
# # 这里省略代理的定义
#
# # 使用内存来在链或调用之间存储状态
# # 这里省略内存的定义
#
# # 测试应用程序
# response = chain.run("the meaning of life")
# print(response)
