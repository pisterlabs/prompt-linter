import logging
from typing import Callable, Any, List
from magic_google import MagicGoogle
import requests
from agent.common import return_parse
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.tools.render import format_tool_to_openai_function, convert_pydantic_to_openai_function
from agent.source.model import GoogleSpiderInput, GoogleSpiderOutput
from base import Base
from agent.source.web import WebSpider
from retrying import retry
import threading


class WebTread(threading.Thread):
    def __init__(self, base: Base, name: str, title: str, url: str):
        logging.info(f"[web thread] start thread {name} {url}")
        threading.Thread.__init__(self)
        self.__name = name
        self.__title = title
        self.__url = url
        self.__spider = WebSpider(base, ChatOpenAI(
            openai_api_key=base.openapi_key,
            openai_api_base=base.openapi_base,
            model_name="gpt-3.5-turbo-16k",
            temperature=0,
        ))
        self.__result = {}

    def run(self) -> None:
        self.__result = self.__spider.search(self.__name, self.__url)

    def get_result(self) -> dict:
        self.__result["ref"] = self.__url
        self.__result["title"] = self.__title
        return self.__result


class GoogleSpider:
    def __init__(self, base: Base, llm: ChatOpenAI):
        self.__base = base
        self.__llm = llm
        self.__google = MagicGoogle([self.__base.proxy])
        self.__session = requests.Session()
        self.__base = base
        self.__llm = llm
        self.__prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个资源查找助手，用户会给出资源关键词，你需要自己提取出资源的名字，然后使用搜索引擎去查找关键词，并对返回的结果进行排序，最有可能包含资源的链接放到最前面。\n"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        self.__tools = [self.__google_search()]
        self.__functions = [format_tool_to_openai_function(t) for t in self.__tools]
        self.__functions.append(convert_pydantic_to_openai_function(GoogleSpiderOutput))
        self.__agent = {
                           "input": lambda x: x["input"],
                           "agent_scratchpad": lambda x: format_to_openai_functions(x['intermediate_steps'])
                       } | self.__prompt | self.__llm.bind(functions=self.__functions) | return_parse(
            "GoogleSpiderOutput",
            {"links": []})
        self.__agent_execute = AgentExecutor(agent=self.__agent, tools=self.__tools, verbose=True)

    @retry(stop_max_attempt_number=3, wait_fixed=2)
    def __google_web_search(self, keyword: str):
        return self.__google.search(keyword)

    def __google_search(self) -> Callable[[str], Any]:
        @tool("googleSpider", args_schema=GoogleSpiderInput)
        def spider(keyword: str) -> list:
            """从谷歌中查找资源"""
            logging.info(f"[google-spider] search keyword {keyword}")
            content = []
            for data in self.__google_web_search(keyword):
                content.append({
                    "title": data["title"],
                    "url": data["url"]
                })
            logging.info(f"[google-spider] spider res is {content}")
            return content

        return spider

    def search(self, name: str) -> list[dict]:
        response = self.__agent_execute.invoke({"input": name}, return_only_outputs=True)
        logging.info(f"[google-spider] response {response}")
        links, threads, result = response.get("links"), [], []
        if len(links) > self.__base.max_size:
            links = links[:self.__base.max_size]
        logging.info(f"[google-spider] links {links}")
        for link in links:
            logging.info(f"[google-spider] start spider {link}")
            thread = WebTread(self.__base, response.get("name"), link["title"], link["url"])
            thread.start()
            threads.append(thread)
        # 等待线程结束
        for thread in threads:
            thread.join()
            info = thread.get_result()
            logging.info(f"[google-spider] get content {info}")
            result.append(info)
        return result
