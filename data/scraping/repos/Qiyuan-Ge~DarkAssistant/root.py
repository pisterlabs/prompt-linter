from typing import List, Union
from langchain import LLMChain
from langchain.llms import OpenAI
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.tools.base import BaseTool
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from fastchat.conversation import get_conv_template

from .helper import get_current_time, get_thoughts_from_intermediate_steps, convert_messages_to_conversation
from .parser import OutputParserForRoot

SPLIT_TOKEN = '<end of turn>'

prompt_template = """\
Current datetime: {date}


You have access to the following tools:
{tools}


Question: the input question you must answer

You should only respond in format as described below:

Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated one or more times)


Here is an example:

Question: Who created Spiderman?
Thought: I need to find out who created Spiderman.
Action: Web Copilot
Action Input: who created Spiderman
Observation: Stan Lee and Steve Ditko created Spider-Man.
Thought: I know who created Spiderman now.
Action: Final Response
Action Input: Stan Lee and Steve Ditko created Spider-Man.
{history}
Now let's answer the following question:

Question: {user}""" + SPLIT_TOKEN + "{agent_scratchpad}"
    
    
# prompt模板
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Union[Tool, BaseTool]]

    def format(self, **kwargs) -> str:
        # 得到中间步骤(AgentAction, Observation tuples)
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = get_thoughts_from_intermediate_steps(intermediate_steps, save_to_file=False)
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{i+1}. {tool.name}: {tool.description}" for i, tool in enumerate(self.tools)])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        chat_history = convert_messages_to_conversation(kwargs["history"])
        kwargs["history"] = chat_history
        kwargs["date"] = get_current_time()

        agent_profile = kwargs.pop("agent_profile")
        template_name = kwargs.pop("template_name")
        conv = get_conv_template(template_name)
        if agent_profile is not None:
            conv.system_message = agent_profile
        conv.sep2 = ''
        
        prompt = self.template.format(**kwargs)
        user_prompt, assistant_prompt = prompt.split(SPLIT_TOKEN)
        
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], assistant_prompt)
        # print(conv.get_prompt())
        return conv.get_prompt()   


def create_root(
    tools: List[Union[Tool, BaseTool]], 
    model_name: str = "text-davinci-003", 
    generate_params: dict = {'max_tokens':1024, 'temperature':0.9, 'top_p':0.6},
    ):
    
    conversation_prompt = CustomPromptTemplate(
        template=prompt_template,
        tools=tools,
        input_variables=["user", "intermediate_steps", "history", "agent_profile", "template_name"]
    )
    
    llm = OpenAI(
        model=model_name,
        temperature=generate_params['temperature'],
        max_tokens=generate_params['max_tokens'],
        top_p=generate_params['top_p'],
        streaming=True, 
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    
    llm_chain = LLMChain(llm=llm, prompt=conversation_prompt) #, callbacks=[handler]
    
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=OutputParserForRoot(),
        stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in tools],
    )
    
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=False)
    
    return agent_executor
