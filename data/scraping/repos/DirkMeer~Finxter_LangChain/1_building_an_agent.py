import re

from decouple import config
from langchain import LLMChain
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
    Tool,
)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import AgentAction, AgentFinish
from prompts import base_agent_template, base_agent_template_w_memory
from tools import MobyDuckSearch, InternetTool


chat_gpt_api = ChatOpenAI(
    temperature=0, model="gpt-3.5-turbo-0613", openai_api_key=config("OPENAI_API_KEY")
)

moby_duck_tool = MobyDuckSearch()

tools = [
    Tool(
        name=moby_duck_tool.name,
        func=moby_duck_tool.run,
        description=moby_duck_tool.description,
    )
]


class MobyDuckPromptTemplate(StringPromptTemplate):
    template: str
    tools: list[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        scratchpad = ""

        for action, tool_output in intermediate_steps:
            scratchpad += action.log
            scratchpad += f"\nObservation: {tool_output}\nThought: "

        kwargs["agent_scratchpad"] = scratchpad
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])

        return self.template.format(**kwargs)


prompt_formatter = MobyDuckPromptTemplate(
    template=base_agent_template,
    tools=tools,
    input_variables=["input", "intermediate_steps"],
)


class MobyDuckOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> AgentAction | AgentFinish:
        if "Final Answer:" in llm_output:
            answer = llm_output.split("Final Answer:")[-1].strip()
            return AgentFinish(
                return_values={"output": answer},
                log=llm_output,
            )

        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2).strip(" ").strip('"')

        return AgentAction(tool=action, tool_input=action_input, log=llm_output)


llm_chain = LLMChain(llm=chat_gpt_api, prompt=prompt_formatter)


moby_duck_agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=MobyDuckOutputParser(),
    stop=["\nObservation:"],
)


agent_executor = AgentExecutor.from_agent_and_tools(
    agent=moby_duck_agent, tools=tools, verbose=True
)


# agent_executor.run("Can you recommend me a zombie game from the year 2022?")


internet_tool = InternetTool()
tools.append(
    Tool(
        name="visit_specific_url",
        func=internet_tool.run,
        description=(
            "Useful when you want more information about a page by opening it's url on the internet."
            "Input should be a valid and full internet url with nothing else attached."
        ),
    )
)


prompt_formatter_w_memory = MobyDuckPromptTemplate(
    template=base_agent_template_w_memory,
    tools=tools,
    input_variables=["input", "intermediate_steps", "history"],
)


llm_chain_w_memory = LLMChain(llm=chat_gpt_api, prompt=prompt_formatter_w_memory)


moby_duck_agent_w_memory = LLMSingleActionAgent(
    llm_chain=llm_chain_w_memory,
    output_parser=MobyDuckOutputParser(),
    stop=["\nObservation:"],
)


memory = ConversationBufferWindowMemory(k=10)


agent_executor_w_memory = AgentExecutor.from_agent_and_tools(
    agent=moby_duck_agent_w_memory, tools=tools, verbose=True, memory=memory
)


agent_executor_w_memory.run("Can you recommend me a zombie game from the year 2022?")
agent_executor_w_memory.run("Can you give me more information on that first game?")
