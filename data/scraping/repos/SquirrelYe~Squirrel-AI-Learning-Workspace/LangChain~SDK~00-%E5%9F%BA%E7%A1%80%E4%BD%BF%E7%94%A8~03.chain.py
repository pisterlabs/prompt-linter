import os
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SimpleSequentialChain, LLMRequestsChain
from langchain.agents import (create_csv_agent, load_tools, initialize_agent, AgentType)
from langchain.callbacks import (get_openai_callback)

os.environ["LANGCHAIN_TRACING"] = "true"


# 使用 LLMChain 生成回复
def use_langchain_chain():
    template = "我的邻居姓{lastname}，他生了个儿子，给他儿子起个名字"
    prompt = PromptTemplate(
        input_variables=["lastname"],
        template=template,
    )
    llm = OpenAI(temperature=0.9)
    chain1 = LLMChain(llm=llm, prompt=prompt)

    # 创建第二条链
    second_prompt = PromptTemplate(
        input_variables=["child_name"],
        template="邻居的儿子名字叫{child_name}，给他起一个小名",
    )
    chain2 = LLMChain(llm=llm, prompt=second_prompt)

    # 链接两条链
    overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)

    # 执行链，只需要传入第一个参数
    catchphrase = overall_chain.run("王")
    print(catchphrase)


# 使用 Agent 生成回复
def use_langchain_agent():
    with get_openai_callback() as cb:
        agent = create_csv_agent(OpenAI(temperature=0), './data.csv', verbose=True)
        agent.run("一共有多少行数据?")
        agent.run("打印一下第一行数据")
        agent.run("将得到的数据保存到文件 ./result.txt 中。")
        print(cb)


# 使用 Agent 获取天气
def use_langchain_agent_weather():
    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent.run("天津有什么好玩儿的地方?")


# 使用 Agent 执行Shell命令
def use_langchain_agent_shell():
    llm = OpenAI(temperature=0)
    tools = load_tools(["shell"], llm=llm)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent.run("ls -l")


# 使用 LLMRequestsChain 的简单案例
def use_langchain_requests_chain():
    with get_openai_callback() as cb:
        template = """在 >>> 和 <<< 之间是网页的返回的HTML内容。
        网页是新浪财经A股上市公司的公司简介。
        请抽取参数请求的信息。

        >>> {requests_result} <<<
        请使用如下的JSON格式返回数据
        {{
        "company_name":"a",
        "company_english_name":"b",
        "issue_price":"c",
        "date_of_establishment":"d",
        "registered_capital":"e",
        "office_address":"f",
        "Company_profile":"g"
        }}
        Extracted:"""

        prompt = PromptTemplate(input_variables=["requests_result"], template=template)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=prompt))
        inputs = {"url": "https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpInfo/stockid/600519.phtml"}
        response = chain(inputs)
        print(response['output'])

        print(cb)

        # {
        #   "company_name":"贵州茅台酒股份有限公司",
        #   "company_english_name":"Kweichow Moutai Co.,Ltd.",
        #   "issue_price":"31.39",
        #   "date_of_establishment":"1999-11-20",
        #   "registered_capital":"125620万元(CNY)",
        #   "office_address":"贵州省仁怀市茅台镇",
        #   "Company_profile":"公司是根据贵州省人民政府黔府函〔1999〕291号文,由中国贵州茅台酒厂有限责任公司作为主发起人,联合贵州茅台酒厂技术开发公司、贵州省轻纺集体工业联社、深圳清华大学研究院、中国食品发酵工业研究院、北京市糖业烟酒公司、江苏省糖烟酒总公司、上海捷强烟草糖酒(集团)有限公司于1999年11月20日共同发起设立的股份有限公司。经中国证监会证监发行字[2001]41号文核准并按照财政部企[2001]56号文件的批复,公司于2001年7月31日在上海证券交易所公开发行7,150万(其中,国有股存量发行650万股)A股股票。"
        # }
        # Tokens Used: 2588
        #         Prompt Tokens: 2203
        #         Completion Tokens: 385
        # Successful Requests: 1
        # Total Cost (USD): $0.005175999999999999


if __name__ == '__main__':
    # use_langchain_chain()
    # use_langchain_agent()
    # use_langchain_agent_weather()
    # use_langchain_agent_shell()
    use_langchain_requests_chain()
