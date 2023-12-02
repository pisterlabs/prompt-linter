from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import Tool, LLMSingleActionAgent, AgentOutputParser, AgentExecutor
from langchain.chains import LLMChain
from typing import List, Union
import re


# This file is used to define the template and output parser of the custom agent in Langchin. Code is consistent with the guidelines in the Langchin documentation.

class AgentBuilder:
    @staticmethod
    def build_agent(tools: List[Tool], template: str, llm, verbose: bool = False):
        class CustomPromptTemplate(StringPromptTemplate):
            template: str

            def format(self, **kwargs) -> str:
                intermediate_steps = kwargs.pop("intermediate_steps")
                thoughts = ""
                for action, observation in intermediate_steps:
                    thoughts += action.log
                    thoughts += f"\nObservation: {observation}\nThought: "
                kwargs["agent_scratchpad"] = thoughts
                kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
                kwargs["tool_names"] = ", ".join([tool.name for tool in tools])
                return self.template.format(**kwargs)

        prompt = CustomPromptTemplate(template=template, tools=tools, input_variables=["input", "intermediate_steps"])

        class CustomOutputParser(AgentOutputParser):
            def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
                if "Final Answer:" in llm_output:
                    return AgentFinish(
                        return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                        log=llm_output,
                    )
                regex = r"Action: (.*?)[\n]*Action Input:[\s]*(.*)"
                match = re.search(regex, llm_output, re.DOTALL)
                if not match:
                    raise ValueError(f"Could not parse LLM output: `{llm_output}`")
                action = match.group(1).strip()
                action_input = match.group(2)
                return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

        output_parser = CustomOutputParser()
        tool_names = [tool.name for tool in tools]
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        agent = LLMSingleActionAgent(llm_chain=llm_chain, output_parser=output_parser, stop=["\nObservation:"], allowed_tools=tool_names)
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose)

        return agent_executor
