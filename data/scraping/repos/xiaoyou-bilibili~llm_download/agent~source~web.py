import logging
from typing import Callable, Any
from bs4 import BeautifulSoup
import cchardet
from magic_google import MagicGoogle
import requests
from agent.common import return_parse
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import tool, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.tools.render import format_tool_to_openai_function, convert_pydantic_to_openai_function
from agent.source.model import WebSpiderInput, WebSpiderOutput
from base import Base
from urllib.parse import urlparse
from retrying import retry


class WebSpider:
    def __init__(self, base: Base, llm: ChatOpenAI):
        self.__base = base
        self.__llm = llm
        self.__google = MagicGoogle([self.__base.proxy])
        self.__session = requests.Session()
        self.__base = base
        self.__llm = llm
        self.__prompt = ChatPromptTemplate.from_messages([
            ("system",
             "你是一个资源下载链接查找助手，用户会提供资源名字和网页链接，你会通过爬虫去爬取网页链接的内容，并从网页内容中找出该资源可能的下载链接或磁力链接并返回给用户"),
            ("user", "资源名字 {name}\n网页链接 {url}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        self.__tools = [self.__web_spider()]
        self.__functions = [format_tool_to_openai_function(t) for t in self.__tools]
        self.__functions.append(convert_pydantic_to_openai_function(WebSpiderOutput))
        self.__agent = {
                           "name": lambda x: x["name"],
                           "url": lambda x: x["url"],
                           "agent_scratchpad": lambda x: format_to_openai_functions(x['intermediate_steps'])
                       } | self.__prompt | self.__llm.bind(functions=self.__functions) | return_parse("WebSpiderOutput",
                                                                                                      {"links": []})
        self.__agent_execute = AgentExecutor(agent=self.__agent, tools=self.__tools, verbose=True)

    @retry(stop_max_attempt_number=3, wait_fixed=2)
    def __web_request(self, url: str) -> requests.Response:
        return self.__session.get(url, proxies=self.__base.proxy, headers=self.__base.http_header)

    def __web_spider(self) -> Callable[[str], Any]:
        @tool("webSpider", args_schema=WebSpiderInput)
        def spider(url: str) -> str:
            """爬取网页信息"""
            logging.info(f"[web-spider] get url {url}")
            resp = self.__web_request(url)
            content = ''
            if resp.status_code == 200:
                parsed_url = urlparse(url)
                resp.encoding = cchardet.detect(resp.content)
                logging.info(f"[web-spider] spider res is {resp.text}")
                # 使用正则去判断一下
                doc = BeautifulSoup(resp.text, 'html.parser')
                # 遍历所有叶子节点，只获取文本
                for tag in doc.find_all(True):
                    text = tag.get_text().strip()
                    # 只遍历子标签
                    if len(tag.find_all(True)) > 0 or text == "":
                        continue
                    if tag.name in ['head', 'script']:
                        continue
                    elif tag.name != 'a':
                        content += f"{text}\n"
                    elif tag is not None and tag.name == 'a':
                        href = tag['href']
                        if href is not None and "http" not in href:
                            href = f"{parsed_url.scheme}://{parsed_url.netloc}{href}"
                        content += f"{text}:{href}\n"
                # logging.info(f"content is {content}")
            return content

        return spider

    def search(self, name: str, url: str) -> dict:
        return self.__agent_execute.invoke({"name": name, "url": url}, return_only_outputs=True)
