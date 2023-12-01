# 参考链接
# https://python.langchain.com/docs/get_started/quickstart

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, BaseOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

llm = OpenAI()
chat_model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

# 现在我们可以开始构建我们的语言模型应用程序。LangChain提供了许多可用于构建语言模型应用程序的模块。模块可以在简单的应用程序中独立使用，也可以组合起来用于更复杂的用例。

# LangChain应用程序的核心构建模块是LLMChain。这结合了三件事：
#   LLM：语言模型是这里的核心推理引擎。为了使用 LangChain，您需要了解不同类型的语言模型以及如何使用它们。
#   提示模板：这为语言模型提供指令。这控制着语言模型的输出内容，因此了解如何构建提示和不同的提示策略至关重要。
#   输出解析器：它们将 LLM 的原始响应转换为更可行的格式，从而可以轻松使用下游的输出。
#   在本入门指南中，我们将单独介绍这三个组件，然后介绍结合了所有组件的 LLMChain。了解这些概念将为您使用和定制 LangChain 应用程序做好准备。大多数 LangChain 应用程序允许您配置 LLM 和/或使用的提示，因此了解如何利用这一点将是一个很大的推动因素。


def generate_llm_reply():
    # 语言模型有两种类型，在LangChain中称为：
    #   LLM：这是一种语言模型，它将字符串作为输入并返回字符串
    #   ChatModels：这是一个语言模型，它将消息列表作为输入并返回消息

    text = "What would be a good company name for a company that makes colorful socks?"
    result1 = llm.predict(text)
    result2 = chat_model.predict(text)
    print(result1 + "\n" + result2)
    # Rainbow Socks Co.
    # ColorfulSox

    messages = [HumanMessage(content=text)]
    result3 = llm.predict_messages(messages)
    result4 = chat_model.predict_messages(messages)
    print(result3)
    print(result4)
    # content='\n\nSockFinity' additional_kwargs={} example=False
    # content='SockSplash' additional_kwargs={} example=False


def generate_with_template():
    # 一般使用
    prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
    temp1 = prompt.format(product="colorful socks")
    print(temp1)
    # What is a good name for a company that makes colorful socks?

    # 结合Chat模型使用
    system_template = "You are a helpful assistant that translates {input_language} to {output_language}."
    human_template = "{text}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    temp2 = chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")
    print(temp2)
    # [
    #   SystemMessage(content='You are a helpful assistant that translates English to French.', additional_kwargs={}),
    #   HumanMessage(content='I love programming.', additional_kwargs={}, example=False)
    # ]


def generate_with_output_parser():
    # OutputParsers 将 LLM 的原始输出转换为可在下游使用的格式。OutputParsers 有几种主要类型，包括：
    #   从 LLM 转换文本 -> 结构化信息（例如 JSON）
    #   将 ChatMessage 转换为字符串
    #   将除消息之外的调用返回的额外信息（如 OpenAI 函数调用）转换为字符串。

    # 继承 BaseOutputParser 类并实现 parse 方法来创建自定义的 OutputParser。
    class CommaSeparatedListOutputParser(BaseOutputParser):
        """Parse the output of an LLM call to a comma-separated list."""

        def parse(self, text: str):
            """Parse the output of an LLM call."""
            return text.strip().split(", ")

    # 测试自定义的 OutputParser
    result = CommaSeparatedListOutputParser().parse("hi, bye")
    print(result)


def generate_demo():
    # 创建一个自定义的 OutputParser
    class CommaSeparatedListOutputParser(BaseOutputParser):

        def parse(self, text: str):
            print("CommaSeparatedListOutputParser ->", text)  # CommaSeparatedListOutputParser -> red, blue, green, yellow, orange
            return text.strip().split(", ")

    # 创建一个自定义的 PromptTemplate
    template = """You are a helpful assistant who generates comma separated lists.
    A user will pass in a category, and you should generated 5 objects in that category in a comma separated list.
    ONLY return a comma separated list, and nothing more."""
    human_template = "{text1} {text2}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # 参数结构：chat_prompt
    # input_variables=['text']
    # output_parser=None
    # partial_variables={}
    # messages=[
    #   SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], output_parser=None, partial_variables={}, template='You are a helpful assistant who generates comma separated lists.\n    A user will pass in a category, and you should generated 5 objects in that category in a comma separated list.\n    ONLY return a comma separated list, and nothing more.', template_format='f-string', validate_template=True), additional_kwargs={}),
    #   HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['text'], output_parser=None, partial_variables={}, template='{text}', template_format='f-string', validate_template=True), additional_kwargs={})
    # ]

    chain = LLMChain(llm=chat_model, prompt=chat_prompt, output_parser=CommaSeparatedListOutputParser())
    result = chain.run({'text1': 'list some colors', 'text2': 'excluding red'})
    result2 = chain.run({'text1': 'list some fruits', 'text2': 'excluding apples'})
    print(result)
    print(result2)
    # ['blue', 'green', 'yellow', 'orange', 'purple']


if __name__ == '__main__':
    # generate_llm_reply()
    # generate_with_template()
    # generate_with_output_parser()
    generate_demo()
