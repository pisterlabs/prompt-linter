# 导入所需的库和模块
import os
from langchain.agents import AgentExecutor, initialize_agent, AgentType, Tool, ConversationalChatAgent
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain import hub
from langchain.utilities.wikipedia import WikipediaAPIWrapper  # 导入WikipediaAPIWrapper

# 相对位置加载配置文件
script_dir = os.path.dirname(os.path.abspath(__file__))
api_key_file_path = os.path.join(script_dir, 'key.txt')

# 自动填写OpenAI API
try:
    with open(api_key_file_path, "r") as key_file:
        api_key = key_file.read().strip()
except FileNotFoundError:
    api_key = input("请输入您的OpenAI API密钥：")


# 定义搜索功能
def search(query):
    wrapper = WikipediaAPIWrapper()
    try:
        output_summaries = wrapper.run(query)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    return output_summaries


# 定义工具列表，包括Current Search工具
tools = [
    Tool(
        name="Current Search",
        func=search,  # 使用您的search函数作为此工具的功能
        description="用于回答有关当前事件或世界当前状态的问题",
    ),
]

# 初始化ChatOpenAI模型，使用从文件或用户输入中获取的API密钥。
chat_model = ChatOpenAI(openai_api_key=api_key, temperature=0.3, model_name="gpt-3.5-turbo", streaming=True)

# 从langchain的hub中获取提示。
prompt = hub.pull("hwchase17/react-chat-json")


# 准备提示。这将处理工具的描述和工具名称，以便模型可以使用它们。
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# 准备模型。这将设置模型的停止标记，以便在观察到特定标记时停止生成输出。
chat_model_with_stop = chat_model.bind(stop=["\nObservation"])


# 准备模板。这将格式化工具的响应，以便模型可以理解和处理它们。
TEMPLATE_TOOL_RESPONSE = """
TOOL RESPONSE:
---------------------
{observation}

USER'S INPUT
--------------------

Okay, so what is the response to my last comment?
If using information obtained from the tools you must mention it explicitly
without mentioning the tool names - I have forgotten all TOOL RESPONSES!
Remember to respond with a markdown code snippet of a json blob with a
single action, and NOTHING else - even if you just want to respond to the user.
Do NOT respond with anything except a JSON snippet no matter what!
"""

# 定义代理管道
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_messages(
            x["intermediate_steps"],  #template_tool_response=TEMPLATE_TOOL_RESPONSE
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | chat_model_with_stop
    | JSONAgentOutputParser()
)


# 初始化内存。这将创建一个用于存储聊天历史的内存对象。
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 创建代理执行器。这将创建一个用于执行代理的对象。
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)

# 在终端中创建一个无限循环，以便用户可以连续输入问题
while True:
    # 提示用户输入问题
    user_input = input("您：")

    # 如果用户输入“退出”，则退出循环
    if user_input.lower() == "退出":
        print("再见！")
        break

    # 将用户输入发送给代理执行器，并获取代理的输出
    response = agent_executor.invoke({"input": user_input})["output"]

    # 在终端中显示代理的输出
    print(f"代理：{response}")