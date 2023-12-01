import warnings

from real_tools.AskUserTool import AskUserTool
from real_tools.opening_phrase_tool import OpeningPhraseTool
from real_tools.search_flight_ticket_tool import SearchFlightTool
from real_tools.search_hotel_tool import SearchHotelTool
from real_tools.search_sight_ticket_tool import SearchSightTicketTool
from real_tools.search_train_ticket_tool import SearchTrainTool

# Filter out the specific UserWarning from the langchain package
warnings.filterwarnings("ignore", message="You are trying to use a chat model. This way of initializing it is no longer supported. Instead, please use: `from langchain.chat_models import ChatOpenAI`")

from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.callbacks import FinalStreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import MessagesPlaceholder, PromptTemplate
from langchain.schema import SystemMessage
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper
import os



from langchain.llms import OpenAI
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "886ab329f3d0dda244f3544efeb257cc077d297bb0c666f5c76296d25c0b2279"




############## chain工具们，prompt是真正定义他们功能的地方 ##############
llm = OpenAI(temperature=0.3)
prompt1 = PromptTemplate(
    input_variables=["content"],
    template="从{content}，分析用户想去哪个城市，分析用户想去哪个城市，必须输出一个具体的departure_city(中文)和一个具体的arrival_city(中文)"
)
prompt2 = PromptTemplate(
    input_variables=["content"],
    template="从{content}，分析用户想什么时候去，必须输出一个2023-11-24以后的日期，输出格式是date:yyyy-MM-dd"
)
prompt4 = PromptTemplate(
    input_variables=["content"],
    template="从{content}，分析用户的喜好，生成搜索的中文关键词"
)
prompt5 = PromptTemplate(
    input_variables=["content"],
    template="把{content}的markdown表格格式的旅行计划表格，补充完整，让最终的旅行计划表变得合理且完整，像一位职业的旅行规划师创作的一样"
)
prompt6 = PromptTemplate(
    input_variables=["content"],
    template="这是一个兜底方法，当对话必须终止，而{content}中的内容不足成为一份旅行计划时，根据{content}补充生成一份旅行计划"
)

search = SerpAPIWrapper()
chain1 = LLMChain(llm=llm, prompt=prompt1)
chain2 = LLMChain(llm=llm, prompt=prompt2)
chain4 = LLMChain(llm=llm, prompt=prompt4)
chain5 = LLMChain(llm=llm, prompt=prompt5)
llm2 = OpenAI(temperature=0.9)
chain6 = LLMChain(llm=llm2, prompt=prompt6)

############## 一个结合了语言模型chain和本地函数tool的tools工具包 ##############
tools = [
    Tool(
        name="网络信息搜索",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions",
    ),
    Tool.from_function(
        func=chain1.run,
        name="热门城市推荐",
        description="根据输入的信息，分析用户想去哪个城市，必须输出departure_city(中文)， arrival_city(中文)",
        # coroutine= ... <- you can specify an async method if desired as well
    ),
    Tool.from_function(
        func=chain2.run,
        name="旅行吉日",
        description="根据输入的信息，获取今天的日期以及用户的偏好，分析什么时间可能适合用户，必须输出一个今天以后的日期，输出格式是yyyy-MM-dd",
        # coroutine= ... <- you can specify an async method if desired as well
    ),
    Tool.from_function(
        func=chain4.run,
        name="用户画像",
        description="根据输入的信息，分析用户的喜好，生成搜索的中文关键词",
        # coroutine= ... <- you can specify an async method if desired as well
    ),
    # Tool.from_function(
    #     func=chain5.run,
    #     name="完善信息",
    #     description="在把final_answer交给用户之前需要用`complete_table`工具补充空白的地方，让最终的旅行计划表变得合理且完整，像一位职业的旅行规划师创作的一样，注意每晚应该住在不同的酒店",
    #     # coroutine= ... <- you can specify an async method if desired as well
    # ),
    Tool.from_function(
        func=chain6.run,
        name="兜底方法",
        description="这是一个兜底方法，当对话必须终止，但还没产生final answer的时候作为降级方案",
        # coroutine= ... <- you can specify an async method if desired as well
    ),
    SearchFlightTool(),
    SearchHotelTool(),
    SearchSightTicketTool(),
    SearchTrainTool(),
    OpeningPhraseTool(),
    AskUserTool(),
]

############## 决策agent，tools只是名字 ##############
prefix = """
            
            一切开始前，先和用户说个开场白
            Answer the following questions as best you can. You have access to the following tools:
            最终目标是生成一份拥有出发和回来时间、交通方式，每天游玩景点，每晚住的酒店的旅行方案,并用`write_file`工具把结果写入到本地文件中"""
suffix = """
    下面是工具的介绍
        `opening_phrase`是你的开场白
        `灵感挖掘`可以向用户发起提问，根据用户的回答进一步做其他推理，如果用户实在没有提供有效答案，可以用`热门城市推荐``旅行吉日`补全信息，但是注意询问用户的次数不能超过四次
        `用户画像`可以分析用户的喜好，拿到搜索条件keyword，
        `Qunar机票搜索`的参数是一个字符串“departure_city + arrival_city + date”中间用“+”分割，
        `Qunar景点搜索`的参数一个字符串，是arrival_city
        `Qunar酒店搜索` 的参数是一个字符串“city + check_in_data + leave_date+ keyword”，中间用“+”分割
        `Qunar火车票搜索`的参数是一个字符“departure_city+  arrival_city+ date”，中间用“+”分割
        你需要确定1.用户准备从哪里出发，去哪里玩
                2.用户什么时间去旅游，准备玩儿几天
                3.用户准备用什么方式去目标地，什么方式回来
                4.用户每天要玩哪些景点
                5.基于这些景点，每晚应该住在哪里，每天住的地方应该不一样
        Begin! 用markdown格式的表格展示你生成的方案，内容是中文，Remember to speak as a pirate when 用markdown格式的表格展示你生成的方案，内容是中文的markdown表格
       类似下面这样，要素一定要齐全
        | 日期   | 出发/回程 | 交通方式        | 价格   | 目的地 | 主要活动      | 住宿        | 酒店价格   |
        |------|-------|-------------|------|-----|-----------|-----------|--------|
        | 7月1日 | 出发    | 飞机（纽约 - 巴黎） | $500 | 巴黎  | 抵达、休息     | 诺富特巴黎中心酒店 | $150/晚 |
        | 7月2日 | -     | -           | -    | 巴黎  | 卢浮宫参观     | 诺富特巴黎中心酒店 | $150/晚 |
        | 7月3日 | -     | -           | -    | 巴黎  | 塞纳河游船     | 诺富特巴黎中心酒店 | $150/晚 |
        | 7月4日 | -     | -           | -    | 巴黎  | 巴黎圣母院、拉丁区 | 诺富特巴黎中心酒店 | $150/晚 |
        | 7月5日 | -     | -           | -    | 巴黎  | 凡尔赛宫一日游   | 诺富特巴黎中心酒店 | $150/晚 |
        | 7月6日 | -     | -           | -    | 巴黎  | 巴黎迪士尼乐园   | 诺富特巴黎中心酒店 | $150/晚 |
        . Use lots of "Args"
    ，让最终的旅行计划表变得合理且完整，像一位职业的旅行规划师创作的一样
        
Question: {input}

History Memory: {memory}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools, prefix=prefix, suffix=suffix, input_variables=["input", "agent_scratchpad"]
)
tool_names = [tool.name for tool in tools]
llm_chain = LLMChain(llm=OpenAI(temperature=0,model_name="gpt-4-1106-preview") #gpt-3.5-turbo-1106 gpt-4-1106-preview
                     , prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)




############## 真正的带了（大脑agent）的执行agent ##############
system_message = SystemMessage(
    # ！！！ 这个system message似乎对结果没影响
    content=""
    # """你是一个旅行搜索小助手，你需要
    #     1、 用 ”guess_city“ 和 ”guess_date“  工具猜到 用户想去的地方和时间 ,
    #     2. 根据1中生成的信息产生departure_city， arrival_city， date这三个参数，然后调用`search_flight`，返回给用户搜索结果
    #     3. 根据1中生成的信息arrival_city，去搜索目标城市的景点
    #     4. 查找景点附近的酒店
    #     4. 把2.3的结果组织成一份旅游规划，返回给用户
    #     6/ 请使用中文回复."""
)
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=300)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}
llm = LLMChain(
    llm=OpenAI(temperature=0
               ,model_name="gpt-4-1106-preview"
               ,streaming=True
               ,callbacks=[FinalStreamingStdOutCallbackHandler(answer_prefix_tokens=["The", "plan", ":"])],)
    , prompt=prompt1)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors = "Check your output and make sure it conforms!", # 并不能生效，去处理错误
    agent_kwargs = agent_kwargs,  # 设定 agent 角色
    memory = memory,  # 配置记忆模式
    #llm=llm,
    max_iterations=10, # ！！！控制action的最大轮数
    early_stopping_method="generate", # !!!兜底策略，超过最大轮数不会戛然而止，会最后调用一次方法,但是目前似乎还没办法精准控制调用哪个
)




if __name__ == '__main__':

    st.title("🔎 Qunar VIVA")
    if prompt := st.chat_input():
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent_executor.run(prompt, callbacks=[st_callback])
            st.write(response)
