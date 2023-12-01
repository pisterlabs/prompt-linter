import pathlib

from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.agents.agent_toolkits import FileManagementToolkit
from langchain.callbacks import FinalStreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
import os

from langchain.tools import ShellTool
os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = "886ab329f3d0dda244f3544efeb257cc077d297bb0c666f5c76296d25c0b2279"

working_directory = pathlib.Path("/")
file_tools = FileManagementToolkit(
    root_dir=str(working_directory.name),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()
tools = FileManagementToolkit(
    root_dir=str(working_directory.name),
    selected_tools=["read_file", "write_file", "list_directory"],
).get_tools()



# # 写入文件
# tools[1]("newfile.txt", "Hello, world!")

############## chain工具们，prompt是真正定义他们功能的地方 ##############
llm = OpenAI(temperature=0.1)


############## 一个结合了语言模型chain和本地函数tool的tools工具包 ##############

############## 决策agent，tools只是名字 ##############
prefix = """一切开始前，先和用户说个开场白
            Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:
            file_tools 中有"read_file", "write_file", "list_directory"这些工具
            最终目标是生成一份拥有出发和回来时间、交通方式，每天游玩景点，每晚住的酒店的旅行方案,并用`write_file`工具把结果写入到/output/example.txt"""
suffix = """
    下面是工具的介绍
            write_file的入参形式为一个字典，其中key是input，input的value是一个json，json的key是"file_path"和 "text"
            
Question: {input}
{agent_scratchpad}"""
prompt = ZeroShotAgent.create_prompt(
    tools, prefix=prefix, suffix=suffix, input_variables=["input", "agent_scratchpad"]
)
shell_tool = ShellTool()
tools.append(shell_tool)
tool_names = [tool.name for tool in tools]
llm_chain = LLMChain(llm=OpenAI(temperature=0,model_name="gpt-3.5-turbo-1106")
                     , prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, ) # 并不能生效，去处理错误)



############## 真正的带了（大脑agent）的执行agent ##############
system_message = SystemMessage(
    # ！！！ 这个system message似乎对结果没影响
    content=""
)
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=300)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}
llm = LLMChain(
    llm=OpenAI(temperature=0
               ,model_name="gpt-3.5-turbo-1106"
               ,streaming=True
               ,callbacks=[FinalStreamingStdOutCallbackHandler(answer_prefix_tokens=["The", "plan", ":"])],)
    , prompt=prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors = "Check your output and make sure it conforms!", # 并不能生效，去处理错误
    agent_kwargs = agent_kwargs,  # 设定 agent 角色
    memory = memory,  # 配置记忆模式
    llm=llm,
    max_iterations=3, # ！！！控制action的最大轮数
    early_stopping_method="generate", # !!!兜底策略，超过最大轮数不会戛然而止，会最后调用一次方法,但是目前似乎还没办法精准控制调用哪个
    toolkit=file_tools
)

print(("用户输入：想下个月去东北玩儿玩儿，我是土豪"))
print(agent_executor.run("想下个月去东北玩儿玩儿，我是土豪"))
