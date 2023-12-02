"""
https://python.langchain.com/docs/modules/chains/

单独使用 LLM 对于简单的应用程序来说很好，但更复杂的应用程序需要链接 LLM - 彼此链接或与其他组件链接。

LangChain为此类“链式”应用提供了Chain接口。

我们将链非常笼统地定义为对组件的调用序列，其中可以包括其他链。


这种将组件组合成链的想法很简单但功能强大

它极大地简化了复杂应用程序的实现并使之更加模块化，从而使调试、维护和改进应用程序变得更加容易。
"""
import env

# LLMChain

"""
Chain PromptTemplate 与 一个 llm
"""

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0.9)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print("chain result: ", chain.run("colorful socks"))

# 多个参数时

prompt = PromptTemplate(
    input_variables=["company", "product"],
    template="What is a good name for {company} that makes {product}?",
)
chain = LLMChain(llm=llm, prompt=prompt)
print("chain result multiple params: ", chain.run({
    'company': "ABC Startup",
    'product': "colorful socks"
    }))

"""
chat model 也可以
"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="What is a good name for a company that makes {product}?",
            input_variables=["product"],
        )
    )
chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
chat = ChatOpenAI(temperature=0.9)
chain = LLMChain(llm=chat, prompt=chat_prompt_template)
print("chain use chat model", chain.run("colorful socks"))