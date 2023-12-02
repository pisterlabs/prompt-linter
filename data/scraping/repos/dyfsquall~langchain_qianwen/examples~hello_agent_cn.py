from langchain.agents import load_tools, AgentExecutor
from langchain_qianwen import Qwen_v1

from langchain_qianwen.agents import ZeroShotAgentCN

if __name__ == "__main__":
    llm = Qwen_v1(
        # model_name="qwen-turbo",
        model_name="qwen-plus",
    )

    tool_names = ["serpapi"]
    tools = load_tools(tool_names)

    custom_agent = ZeroShotAgentCN.from_llm_and_tools(llm=llm, tools=tools)
    agent_exector = AgentExecutor.from_agent_and_tools(
        agent=custom_agent, tools=tools, verbose=True
    )

    agent_exector.run("福岛最近发现哥斯拉了吗?")
