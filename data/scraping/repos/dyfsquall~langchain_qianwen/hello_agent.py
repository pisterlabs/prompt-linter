from langchain.agents import load_tools, AgentType, initialize_agent
from langchain_qianwen import Qwen_v1

if __name__ == "__main__":
    llm = Qwen_v1(
        model_name="qwen-plus",
        # temperature=0.1,
    )
    
    tool_names = ["serpapi"]
    tools = load_tools(tool_names)
    agent = initialize_agent(tools=tools,
                             llm=llm,
                             agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                             verbose=True)
    agent.run("深圳今天出门用带雨伞吗?")
