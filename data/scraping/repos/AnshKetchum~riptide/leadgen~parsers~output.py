from langchain.agents import AgentOutputParser, Agent
from typing import Union, List 
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re 

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        
        print('out', regex, match, llm_output)
        
        if not match:

            regex2 = r"Action\s*\d*\s*:(.*?)\n*Input\s*\d*\s*:[\s]*(.*)"
            match = re.search(regex2, llm_output, re.DOTALL)

            if not match:
                raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        
        
        action = match.group(1).strip()
        action_input = match.group(2)

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
output_parser = CustomOutputParser()