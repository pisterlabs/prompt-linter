from langchain.agents import AgentType
from langchain.agents.agent_toolkits import create_csv_agent

from langchain_qianwen import Qwen_v1


if __name__ == "__main__":
    llm = Qwen_v1(
        model_name="qwen-turbo",
        # model_name="qwen-plus",
    )

    csv_agent_executer = create_csv_agent(
        llm=llm,
        path="./assets/episode_info.csv",
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        )

    # 查询 csv 文件中的信息
    csv_agent_executer.run(
        # """which writer wrote the most episodes? includes the partial participation how many episodes did he write?"""
        """print seasons ascending order of the number of episodes in each season"""
        )
