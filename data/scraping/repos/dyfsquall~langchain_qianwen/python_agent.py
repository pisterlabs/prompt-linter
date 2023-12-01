from langchain.agents import AgentType
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools import PythonREPLTool

from langchain_qianwen import Qwen_v1


if __name__ == "__main__":
    llm = Qwen_v1(
        model_name="qwen-turbo",
        # model_name="qwen-plus",
    )

    python_agent_executer = create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        )

    # 编写python脚本生成 知乎二维码
    python_agent_executer.run(
        # follow the example code {GENERATE_QRCODE_TEMPLATE}, 
        """use qrcode library generate a ORcode that point to www.zhihu.com and save in current working directory.
        """
        )