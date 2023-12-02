from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# openai key 的配置在环境变量中完成 export OPENAI_API_KEY="..."
# 创建一个 openai 的装饰器
llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo-0613")

# result = llm.predict("What would be a good company name for a company that makes colorful socks?")
# print("predict result:", result)

# 对话模式
chat = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo-0613")

result = chat.predict_messages([SystemMessage(content="一个网络安全专家"), \
                                HumanMessage(content="如何查询本机暴露的端口")])
# HumanMessage(content="一个老人在路边卖蔬菜，只赚了二十块钱，市场监督说老人售卖的蔬菜是不合格的，将要罚款2万块，如何评价这件事")])
print("网络安全专家模式输出:\n", result.content)

# 模板，就是占位符替换
print("------------------------------------")
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("帮我介绍一下这个工具的使用技巧 {product}?")
promptFormatted = prompt.format(product="colorful socks")
print("模板 输出: ", promptFormatted)

# Chain 就是连一块喽
print("------------------------------------")
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
chainResult = chain.run("Nmap")
print("chainResult: ", chainResult)

# agent
# 为了处理复杂的工作流，我们需要能够根据输入动态地选择行为
# agent 做的仅仅是：使用语言模型去判断需要做的行为与行为的顺序。
# agent 提供了访问工具的能力，并且可以重复选择一个工具、运行工具，直到过去一个最终的结果，将结果显示。

# 为了获取一个agent，我们需要选择：
# 1. LLM: 一个大语言模型
# 2. Tools: 一个指定自责的方法，比如：Google 搜索，查看数据库，Python REPL 等等。https://python.langchain.com/docs/modules/agents/tools/
# 3. agent name：一个字符串引用到一个已经支持的 agent class。agent class 主要由语言模型来确定要采取哪个操作的提示来参数化。
#                由于本笔记本重点关注最简单、最高级别的 API，因此这仅涵盖使用标准支持的 agent。
#                你可以自定义 agent： https://python.langchain.com/docs/modules/agents/how_to/custom_agent.html

# 下面是一个利用搜索引擎 agent 的例子
print("\n")
print("------------------------------------")
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

# The language model we're going to use to control the agent.
llm = OpenAI(temperature=0)

# The tools we'll give the Agent access to. Note that the 'llm-math' tool uses an LLM, so we need to pass that in.
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Let's test it out!
agentResult = agent.run("最近全")

print("搜索引擎 agent 结果:", agentResult)

# memory
# 连续对话
print("\n")
print("------------------------------------")

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "The following is a friendly conversation between a human and an AI. The AI is talkative and "
        "provides lots of specific details from its context. If the AI does not know the answer to a "
        "question, it truthfully says it does not know."
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

llm = ChatOpenAI(temperature=0)
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)

one = conversation.predict(input="Hi there!")
print("one: ", one)
two = conversation.predict(input="I'm doing well! Just having a conversation with an AI.")
print("two: ", two)
three = conversation.predict(input="Tell me about yourself.")
print("three: ", three)