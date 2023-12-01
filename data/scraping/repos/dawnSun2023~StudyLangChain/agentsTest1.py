"""
LangChain中的链描述了将LLM与其他组件结合起来完成一个应用程序的过程，这也是LangChain名字的由来。
"""
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain,SimpleSequentialChain
import os
from langchain.agents import create_csv_agent,load_tools,initialize_agent,AgentType
from dotenv import dotenv_values

# 读取 .env 文件
env_vars = dotenv_values('.env')

#OPENAI_API_KEY地址：https://platform.openai.com/account/api-keys
os.environ["OPENAI_API_KEY"] = "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"
# os.environ["OPENAI_API_KEY"] = env_vars.get('OPENAI_API_KEY')

#serpapi的API地址：https://serpapi.com/manage-api-key
os.environ["SERPAPI_API_KEY"] = "205c455d1e791d755e759ee92ec13d025204bdabd979dcae4375f17dd0b394fb"
# os.environ["SERPAPI_API_KEY"] = env_vars.get('SERPAPI_API_KEY')
#案例一：
template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"
prompt = PromptTemplate(
    input_variables=["lastname"],
    template=template,
)
llm = OpenAI(temperature=0.9)
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("王"))
print("--------------------------------------------------")
#案例二：第一个模型的输出，作为第二个模型的输入，可以使用LangChain的SimpleSequentialChain
template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"

prompt = PromptTemplate(
    input_variables=["lastname"],
    template=template,
)
llm = OpenAI(temperature=0.9)

chain = LLMChain(llm = llm,
                  prompt = prompt)
# 创建第二条链
second_prompt = PromptTemplate(
    input_variables=["child_name"],
    template="邻居的儿子名字叫{child_name}，给他起一个小名",
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)
# 链接两条链
overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

# 执行链，只需要传入第一个参数
catchphrase = overall_chain.run("王")
print("--------------------------------------------------")
"""
代理
尽管大语言模型非常强大，它们也有一些局限性，它们不能回答实时信息，它们没有上下文的概念，
导致无法保存状态，它们处理数学逻辑问题仍然非常初级，我们只能借助于第三方的工具来完成这些需求，
比如使用搜索引擎或者数据库，LangChain中代理的作用就是根据用户需求，来去访问这些工具。
我们先明确几个概念：
代理负责控制整段代码的逻辑和执行，代理暴露了一个接口，用来接收用户输入，并返回AgentAction或AgentFinish。
AgentAction 决定使用哪个工具
AgentFinish 意味着代理的工作完成了，返回给用户结果
工具
第三方服务的集成，比如谷歌、bing等等，后面有详细列表
工具包
一些集成好了代理包，比如create_csv_agent 可以使用模型解读csv文件，代码如下：
"""
# agent = create_csv_agent(OpenAI(temperature=0), 'data.csv', verbose=True)
# str = agent.run("一共有多少行数据?")
# print(str)
# print("--------------------------------------------------")
#案例三
llm = OpenAI(temperature=0.9)
tools = load_tools(["serpapi"],llm=llm)
agent = initialize_agent(tools, llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("明天在北京穿什么衣服合适?")

"""
LangChain现在支持的工具如下：
Apify	一个数据抓取平台
ArXiv	arXiv是一个收集物理学、数学、计算机科学、生物学与数理经济学的论文预印本的网站
AWS Lambda	Amazon serverless计算服务
Shell工具	执行shell命令
Bing Search	Bing搜索
ChatGPT插件	
DuckDuckGo	DuckDuckGo搜索
Google Places	Google地点
Google Search	Google搜索
Google Serper API	一个从google搜索提取数据的API
Gradio Tools	Gradio应用
IFTTT Webhooks	一个新生的网络服务平台，通过其他不同平台的条件来决定是否执行下一条命令
OpenWeatherMap	天气查询
Python REPL	执行python代码
Requests	发送网络请求
SceneXplain	一个访问ImageCaptioning的工具，通过url就可以获取图像描述
Wikipedia	查询wiki数据
Wolfram Alpha	一个计算平台，可以计算复杂的数学问题
YouTubeSearchTool	视频搜索
Zapier	一个工作流程自动化平台
"""