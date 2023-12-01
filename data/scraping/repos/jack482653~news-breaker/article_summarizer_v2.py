from langchain import PromptTemplate
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.chat_models.openai import ChatOpenAI

from news_breaker.utils.browser import Browser

TEMPLATE = """
1. 根據連結使用 selector: article || .fncnews-content || .story, attributes: [innerText] 取得 innerText
2. 以台灣慣用詞彙產生100字以內的繁體中文總結。

連結：{url}
"""


class ArticleSummarizer:
    def __init__(self, url):
        self.url = url

    def get_prompt(self) -> str:
        prompt = PromptTemplate(
            input_variables=["url"],
            template=TEMPLATE,
        )

        return prompt.format(url=self.url)

    async def summarize(self) -> str:
        result = ""
        async with Browser() as browser:
            # Initialize playwright toolkit
            toolkit = PlayWrightBrowserToolkit.from_browser(
                async_browser=browser.get_browser()
            )
            tools = toolkit.get_tools()
            tools_by_name = {tool.name: tool for tool in tools}
            navigate_tool = tools_by_name["navigate_browser"]
            get_elements_tool = tools_by_name["get_elements"]
            # Initialize agent and model
            chat = ChatOpenAI(temperature=0)
            agent_chain = initialize_agent(
                [navigate_tool, get_elements_tool],
                chat,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
            )
            # Run agent
            result = await agent_chain.arun(self.get_prompt())

        return result
