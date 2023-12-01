"""
本演练展示了如何使用代理来实现专门用于文档存储的 ReAct 逻辑。
"""

import env

from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer

docstore = DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="useful for when you need to ask with search",
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="useful for when you need to ask with lookup",
    ),
]

llm = OpenAI(temperature=0, model_name="text-davinci-002")
react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)

question = "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?"
react.run(question)