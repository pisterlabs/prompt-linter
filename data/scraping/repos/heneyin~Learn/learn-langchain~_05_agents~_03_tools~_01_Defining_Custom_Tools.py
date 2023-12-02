"""
https://python.langchain.com/docs/modules/agents/tools/custom_tools

一个 tool 的组成部分：
1. name: 必须，tool 的唯一名字，字符串
2. description: 非必须，tool 描述，
3. return_direct(bool),defaults to False
4. args_schema((Pydantic BaseModel)),optional,can be used to provide more information (e.g., few-shot examples) or validation for expected parameters.

"""

"""
两种定义方式
"""

import env

# Import things that are needed generically
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool

llm = ChatOpenAI(temperature=0)

"""
Completely New Tools
String Input and Output

最简单的是接受一个单独的查询串，返回一个 string，注意，只有一个输入。多个输入可以看下面的 StructuredTool。

两种方式：直接使用 Tool dataclass 或者通过 BaseTool class 的子类。
"""

"""
'Tool' dataclass 包装了一个func，输入输出都是一个字符串
"""


def noop(input):
    return '3000年'


tools = [
    Tool.from_function(
        func=noop,
        name="search",
        description="useful for when you need to answer questions about current events or current date"
        # coroutine= ... <- you can specify an async method if desired as well
    ),
]

"""
You can also define a custom `args_schema`` to provide more information about inputs.

"""

from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    question: str = Field()


llm_math_chain = LLMMathChain(llm=llm, verbose=True)

tools.append(
    Tool.from_function(
        func=llm_math_chain.run,
        name="Calculator",
        description="useful for when you need to answer questions about math",
        args_schema=CalculatorInput
        # coroutine= ... <- you can specify an async method if desired as well
    )
)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.run(
    "现在是几几年, 年份乘以34是多少?",
)
