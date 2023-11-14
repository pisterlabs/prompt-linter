import re
import json
import time
import requests
from typing import List, Union
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools.base import BaseTool
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from fastchat.conversation import get_conv_template

SPLIT_TOKEN = '<end of turn>'

prompt_template = """Current datetime is {date}

You have access to the following tools:
{tools}


Question: the input question you must answer

You should only respond in format as described below:

Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: {{"arg name": "value"}}
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated one or more times)


Here are two examples:
1. Task that require tools

{example}

2. Task that DON'T require tools(such as daily conversation, straightforward tasks)

Question: Hi.
Thought: I should greet the user.
Action: Final Response
Action Input: Hello! How can I assist you today?
{history}
Now let's answer the following question:


Question: {user}""" + SPLIT_TOKEN + "{agent_scratchpad}"


def get_current_time():
    return time.strftime('%c')


def get_current_city():
    try:
        response = requests.get("http://ipinfo.io")
        data = response.json()
        city = data.get("city", "unknown")
        return city
    except Exception:
        return "unknown"


def convert_messages_to_conversation(messages: List[dict]) -> str:
    if len(messages) == 0:
        return "\n"
    else:
        conv_prompt = "\n\nPrevious chat history:\n{content}\n\n"
        content = ""
        for _, message in enumerate(messages):
            if message["role"] == "user":
                content += f"user: {message['content']}\n"
            elif message["role"] == "assistant":
                content += f"assistant: {message['content']}\n"
        
        return conv_prompt.format(content=content.strip())
    
    
# prompt模板
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Union[Tool, BaseTool]]

    def format(self, **kwargs) -> str:
        # 得到中间步骤(AgentAction, Observation tuples)
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            # thoughts += f"\nObservation: {observation}\n"
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{i+1}. {tool.name}: {tool.description}" for i, tool in enumerate(self.tools)])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        chat_history = convert_messages_to_conversation(kwargs["history"])
        kwargs["history"] = chat_history
        kwargs["date"] = get_current_time()
        kwargs["city"] = get_current_city()

        system_message = kwargs.pop("system_message")
        conv_template_name = kwargs.pop("conv_template_name")
        conv = get_conv_template(conv_template_name)
        if system_message is not None:
            conv.system_message = system_message
        conv.sep2 = ''
        
        origin_prompt = self.template.format(**kwargs)
        user_prompt, assistant_prompt = origin_prompt.split(SPLIT_TOKEN)
        
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], assistant_prompt)
        print(conv.get_prompt())
        return conv.get_prompt()   
    

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # 解析action和action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        # 检查是否应该停止
        if action == "Final Response":
            response = action_input
            return AgentFinish(
                return_values={"output": response},
                log=llm_output,
            )
        else:
            # 定义正则表达式模式，匹配包含键值对的花括号结构
            pattern = r'\{[^{}]*\}'
            # 使用正则表达式进行匹配
            action_input = re.findall(pattern, action_input)[0]
            action_input = json.loads(action_input)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)


def load_agent(model_name: str, tools: List[Union[Tool, BaseTool]], generate_params: dict):
    
    conversation_prompt = CustomPromptTemplate(
        template=prompt_template,
        tools=tools,
        input_variables=["user", "intermediate_steps", "example", "history", "system_message", "conv_template_name"]
    )
    
    llm = OpenAI(
        model=model_name,
        temperature=generate_params['temperature'],
        max_tokens=generate_params['max_tokens'],
        top_p=generate_params['top_p'],
        streaming=True, 
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    
    llm_chain = LLMChain(llm=llm, prompt=conversation_prompt)
    
    output_parser = CustomOutputParser()
    
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in tools],
    )
    
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    
    return agent_executor
