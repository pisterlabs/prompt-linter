# 本质上就是一个 Agent 的流程
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-4', temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm) # serpapi => 谷歌搜索

print(f"工具名称:{tools[1].name},工具作用:{tools[1].description}")

agent = initialize_agent( # 初始化 agent
    tools, # 定义 agent 的 tools
    llm,
    agent="zero-shot-react-description",
    verbose=True # 🔥🔥 显示思考过程
)

print(agent.agent.llm_chain.prompt.template)
"""
	Answer the following questions as best you can. You have access to the following tools:

	Search: A search engine. Useful for when you need to answer questions about current events. Input should be a search query.
	Calculator: Useful for when you need to answer questions about math.

	Use the following format:

	Question: the input question you must answer
	Thought: you should always think about what to do
	Action: the action to take, should be one of [Search, Calculator]
	Action Input: the input to the action
	Observation: the result of the action
	... (this Thought/Action/Action Input/Observation can repeat N times)
	Thought: I now know the final answer
	Final Answer: the final answer to the original input question

	Begin!

	Question: {input}
	Thought:{agent_scratchpad}
"""

agent.run("乔布斯十年后多少岁？")

"""
	,我需要知道乔布斯的出生年份和他去世的年份，然后我可以计算出他如果还活着，十年后会多少岁。
	Action: Search
	Action Input: 乔布斯 出生年份^R

	Observation: February 24, 1955, San Francisco, CA
	Thought:乔布斯的出生年份是1955年。现在我需要知道他去世的年份。
	Action: Search
	Action Input: 乔布斯 去世年份
	Observation: 1985年，因公司内部权力斗争离开苹果公司，后创办皮克斯动画工作室。 1997年，乔布斯回归苹果任职；次年推出iMac，带领苹果度过财务危机。 2011年8月24日，乔布斯向苹果董事会辞去首席执行官职务；同年10月5日，因胰腺神经内分泌肿瘤逝世，享年56岁。
	Thought:乔布斯在2011年去世，享年56岁。现在我可以计算出如果他还活着，十年后会多少岁。
	Action: Calculator
	Action Input: 2011 - 1955 + 56 + 10
	Observation: Answer: 122
	Thought:我现在知道了最后的答案。
	Final Answer: 如果乔布斯还活着，十年后他将会是122岁。
"""


# 更复杂的 agent 可以参考 baby agi