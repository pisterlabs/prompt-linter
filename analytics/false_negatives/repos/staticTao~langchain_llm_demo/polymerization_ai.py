# 以下代码现在能同时实现三个agent的调用。【openai调用数据库，调用自定义api接口查询数据，调用自定义邮件系统等】。
# 成功率在70%
# 打造多链调用，目前在以前版本的成功率稍高
# langchain-0.0.263
# langchain_experimental-0.0.8
# 使用步骤：
#       1.如果需要调用自定义api。请自行编写接口。可以参照demo_api文件夹,采用java形式
#       2.使用数据库查询 需要修改f"mysql+pymysql://root:123456@192.168.20.105/test"。
#       3.添加openai_key


import gradio as gr
import inspect
import json
import re
import socket
from typing import List, Union
import textwrap
import time

import requests
from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain, SQLDatabase
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.prompts import PromptTemplate

from langchain.llms.base import BaseLLM
from langchain_experimental.sql import SQLDatabaseChain

# 这里的key最好是付费的，不然有每分钟请求限制。所以才写了很多个key，每个工具里面可以稍加改装，放入不同的key
key2 = 'sk-xxx'  # key2在query_details使用，里面有解释为什么要用key2
key4 = 'sk-xxx'
# 数据库链接
url = f"mysql+pymysql://root:123456@192.168.20.105/test"

# 远程调用函数的具体方法
function_descriptions = [
    {
        "name": "query_details_api",
        "description": "关于xx的信息在此",
        "parameters": {
            "type": "object",
            "properties": {
                "keyWorks": {
                    "type": "string",
                    "description": "这里是xx/xx/xx/xx中的某一种数据(日期为yyyy-mm-dd格式)",
                }
            },
            "required": ["keyWorks"],
        }

    },
    {
        "name": "ToEmailAPI",
        "description": "发送邮件函数",
        "parameters": {
            "type": "object",
            "properties": {
                "email_content": {
                    "type": "string",
                    "description": "邮件内容",
                },
                "recipient_email": {
                    "type": "string",
                    "description": "收件人邮箱地址",
                }
            },
            "required": ["email_content", "recipient_email"],
        }
    }
]

# tool中使用QA的通用模板
CONTEXT_QA_TMPL = """
根据以下提供的信息，回答用户的问题
信息：{context}

问题：{query}

"""
CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)


def get_function_parameter_names(function):
    if function is not None and inspect.isfunction(function):
        parameter_names = inspect.signature(function).parameters.keys()
        return list(parameter_names)
    else:
        return None


# 自定义api查询
def query_scrap_details_api(keyWorks):
    localIP = socket.gethostbyname(socket.gethostname())  # 这个得到本地ip
    url = 'http://' + localIP + ':8081/xxxx?keyWorks=' + keyWorks
    print(url)
    response = requests.get(url)
    return response.text


# 自定义api 打造邮件系统
def send_email_api(recipientEmail, content):
    localIP = socket.gethostbyname(socket.gethostname())  # 这个得到本地ip
    headers = {
        'Content-type': 'application/json',
        'Accept': 'application/json'
    }
    url = 'http://' + localIP + ':8081/sendEmail'
    data = {
        "recipientEmail": recipientEmail,
        "content": content
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        print(response.text)
    except Exception as e:
        print("发送邮件失败！")


# 增加流式输出
def output_response(response: str) -> None:
    if not response:
        exit(0)
    for line in textwrap.wrap(response, width=60):
        for word in line.split():
            for char in word:
                print(char, end="", flush=True)
                time.sleep(0.1)  # 在每个字符之间增加0.1秒的延迟
            print(" ", end="", flush=True)  # 在每个单词之间加一个空格
        print()  # 在每行打印后移动到下一行
    print("----------------------------------------------------------------")


# tools集合类
class DataSource:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def send_email(self, query: str) -> str:
        """发送邮件"""
        email_addr = ''
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        match = re.search(pattern, query)
        if match:
            print(match.group())
            email_addr = match.group()

        index = query.find("content")
        conent = query[index + 9:]
        send_email_api(email_addr, conent)

    # 采用openai函数方法
    def query_details(self, query: str) -> str:
        """查询xx详情"""
        # 这里不太清楚怎么回事。如果用llm的话，就会报错。尝试了多次。只有换一个实例对象才不报错
        llm1 = ChatOpenAI(model="gpt-3.5-turbo-0613", openai_api_key=key2)

        ai_history = []
        parameter_names = get_function_parameter_names(query_scrap_details_api)

        first_response = llm1.predict_messages([HumanMessage(content=query)], functions=function_descriptions)

        ai_history.append(first_response.additional_kwargs)

        function_name = first_response.additional_kwargs["function_call"]["name"]
        arguments = json.loads(first_response.additional_kwargs["function_call"]["arguments"])

        the_function = globals().get(function_name)
        parameter_names = get_function_parameter_names(the_function)
        parameter_values = []
        for parameter_name in parameter_names:
            parameter_values.append(arguments[parameter_name])

        returned_value = the_function(*parameter_values)
        print("===========================returned_value===========================")
        print(returned_value)
        print("====================================================================")

        return self.llm(str(returned_value))


# 此处在 最好是写出一定的例子。让模型更好的仿造
AGENT_TMPL = """
你是一家顶级工业制造公司中才华横溢的数据分析师，你需要做的工作的是分析用户的行为并做出自己的思考。
请时刻记住你的身份，因为这些数据只能拥有这个身份的人做，这个身份非常重要，请牢记你是数据分析师。

按照给定的格式回答以下问题。你可以使用下面这些工具：
每一次思考尽可能全面，要充分利用以下工具。
{tools}

回答时需要遵循以下用---括起来的示例：

---
Question: 我需要回答的问题
Thought: 回答这个上述我需要做些什么
Action: '{tool_names}' 中的其中一个工具名
Action Input: 选择工具所需要的输入
Observation: 选择工具返回的结果（不要修改结果数据，确保数据的准确性）
...（这个思考/行动/行动输入/观察可以重复N次）
Thought: 我现在知道最终答案
Final Answer: 原始输入问题的最终答案

参考一：
Question: 2023年7月5日有xxxx，其中xxxxx最高是多少？他的操作者是谁？联系电话是多少？
Thought: 需要利用工具查询xx信息，找到xxx最高的数据和操作者.
Action: 查询xx详情
Action Input: 2023-07-05
Observation: 找到 xxx 和 create_name 字段的结果
Thought: 利用工具查询到人员详细信息中找到判定人的信息
Action: 人员详细信息
Action Input: 张三
Observation:
            张三的信息如下：
            - 创建时间：这是时间
            - 性别：这是性别
            - 电话：这是电话
            - 员工编号：这是员工编号
            - 部门：这是部门
            - 家庭地址：这是家庭住址
            - 身份证号码：这是身份证号码
            - 岗位名称：这是岗位名称
            - 邮箱：这是邮箱
            找到 Question中的某些字段进行返回.
Thought: 我现在知道2023年7月5日的xx信息和操作者的电话.
Final Answer: 2023年7月5日xxxx,其中xxx最高是5%,xxxx数据的人是张三，他的联系电话是1888888。
---

现在开始回答，记得在给出最终答案前多按照指定格式进行一步一步的推理。
如果你认为在之前的对话中已经有足够的信息，可以参考之前的对话，直接做出回答。
{chat_history}
Question: {input}
{agent_scratchpad}

"""


# 构建统一的Prompt模板
class CustomPromptTemplate(StringPromptTemplate):
    template: str  # 标准模板
    tools: List[Tool]  # 可使用工具集合

    def format(self, **kwargs) -> str:
        """
        按照定义的 template，将需要的值都填写进去。

        Returns:
            str: 填充好后的 template。
        """
        intermediate_steps = kwargs.pop("intermediate_steps")  # 取出中间步骤并进行执行
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts  # 记录下当前想法
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )  # 枚举所有可使用的工具名+工具描述
        kwargs["tool_names"] = ", ".join(
            [tool.name for tool in self.tools]
        )  # 枚举所有的工具名称
        cur_prompt = self.template.format(**kwargs)
        print(cur_prompt)
        return cur_prompt


# 构建统一的输出解析器
class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        """
        解析 llm 的输出，根据输出文本找到需要执行的决策。
        """
        if "Final Answer:" and "如上" in llm_output in llm_output:  # 如果句子中包含 如上 则代表有简略信息。
            return AgentFinish(
                return_values={"output": llm_output.split("Thought:")[-1].strip()},
                log=llm_output,
            )
        elif "Final Answer:" in llm_output:  # 如果句子中包含 Final Answer 则代表已经完成
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"  # 解析 action_input 和 action
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            return llm_output
        action = match.group(1).strip()
        action_input = match.group(2)

        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(elem_id="chat-box", show_label=False).style(height=750)
    msg = gr.Textbox(placeholder="请输入问题，按回车确认！")


    def respond(message, chat_history):
        try:
            response = agent_executor.run(message)
            # 将用户输入和回答添加到聊天历史中
            chat_history.append((message, response))
            return "", chat_history
        except Exception as e:
            print(e)
            print(f'error file:{e.__traceback__.tb_frame.f_globals["__file__"]}')
            print(f"error line:{e.__traceback__.tb_lineno}")
            chat_history.append((message, "暂未查找到，请重试！"))
            return "", chat_history


    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    llm = OpenAI(temperature=0,
                 openai_api_key=key4,
                 model_name="gpt-3.5-turbo-0613")

    db = SQLDatabase.from_uri(url)
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
    data_source = DataSource(llm)
    tools = [

        Tool(
            name="人员详细信息",
            func=db_chain.run,
            description="当用户询问公司人员信息的时候，可以通过这个工具包进行查找（包括人员的总数）注意：每次查询的sql语句用’selecct * from table‘，不要自己选取列名查询。思考时需要联系上下文，并且不能只想到姓名，要想到这个人的所有信息。对于查询结果需要做中文替换【真实姓名：name，电话号码：telephone，身份证：id_number，家庭住址：home_address，岗位名称：post_name，部门：department】",
        ),
        Tool(
            name="查询xxx详情",
            func=data_source.query_details,
            description="需要知道xxx详情的时候，可以通过这个工具包进行操作",
        ),
        Tool(
            name="发送邮件系统",
            func=data_source.send_email,
            description="发送邮件时，请使用这个工具。传入参数为接收人邮箱（recipientEmail）和发送内容（content）。例：\n"
                        "给张三发送一封关于xxx的邮件\n"
                        "{"
                        "   recipientEmail: 122222@qq.com,"
                        "   content:今天xxxxx"
                        "}",
        )
    ]

    output_parser = CustomOutputParser()

    agent_prompt = CustomPromptTemplate(
        template=AGENT_TMPL,
        tools=tools,
        input_variables=["input", "chat_history", "intermediate_steps"],
    )

    # 添加记忆功能
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(llm=llm, prompt=agent_prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )

    demo.launch(server_name='127.0.0.1', server_port=7866, show_api=False, share=False, inbrowser=False)
