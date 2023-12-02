import json

from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from news_breaker.models.loggable import Loggable
from news_breaker.utils.browser import Browser

TEMPLATE = """
將以下文章總結成最多100字：

{article}
"""


class ArticleSummarizer(metaclass=Loggable):
    def __init__(self, url):
        self.url = url

    def get_prompt(self, article) -> list:
        chat_prompt = ChatPromptTemplate.from_messages(
            [HumanMessagePromptTemplate.from_template(TEMPLATE)]
        )

        return chat_prompt.format_prompt(article=article).to_messages()

    async def summarize(self) -> str:
        result = ""
        self.__logger.info(f"[START] Summarize {self.url}")
        async with Browser() as browser:
            toolkit = PlayWrightBrowserToolkit.from_browser(
                async_browser=browser.get_browser()
            )
            tools = toolkit.get_tools()
            tools_by_name = {tool.name: tool for tool in tools}
            navigate_tool = tools_by_name["navigate_browser"]
            get_elements_tool = tools_by_name["get_elements"]

            self.__logger.info(f"Navigating to {self.url}...")
            await navigate_tool.arun({"url": self.url})
            self.__logger.info("Getting article text...")
            text = await get_elements_tool.arun(
                {"selector": "article, .fncnews-content, .story"}
            )
            self.__logger.info(f"Retrieved article text: {text}")

            selected = json.loads(text)
            articles = map(lambda x: x["innerText"], selected)
            articles_str = "\n".join(articles)

            chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613", temperature=0)
            prompt = self.get_prompt(articles_str)
            self.__logger.info(f"Prompt: {prompt}")
            result = chat(self.get_prompt(articles_str)).content

        self.__logger.info(f"[END] Summarize {self.url}, result: {result}")

        return result
