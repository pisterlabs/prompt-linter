import re
import datetime
from pydantic import BaseModel
from typing import Tuple
# from llm_agents.llm import ChatLLM
from typing import Dict, List
from langchain.schema import ChatMessage
from langchain.agents import Tool
from paperai.llm import *
from paperai import paperai
from paperai import prompt

class CustomAgent(BaseModel):
    # The large language model that the Agent will use to decide the action to take
    chatllm: ChatLLM
    # The prompt that the language model will use and append previous responses to
    prompt: str
    # The list of tools that the Agent can use
    tools: List[Tool]
    # Adjust this so that the Agent does not loop infinitely
    max_loops: int = 5
    # The stop pattern is used, so the LLM does not hallucinate until the end
    stop_pattern: List[str]
    # Pdf details
    pdf_details: Dict[str,str] = {}

    @property
    def tool_by_names(self) -> Dict[str, Tool]:
        return {tool.name: tool for tool in self.tools}
    
    @property
    def tool_description(self) -> str:
        return "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
    
    def run(self, question: str):
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        previous_responses = []
        num_loops = 0
        while num_loops < self.max_loops:
            num_loops += 1
            curr_prompt = self.prompt.format(previous_responses=('\n'.join(previous_responses)))
            output, tool, tool_input = self._get_next_action(curr_prompt)
            if tool == 'Final Answer':
                return tool_input
            tool_result = name_to_tool_map[tool].run(tool_input)
            if tool_result:
                full_result = tool_result.split("***%")
                if len(full_result) > 1:
                    self.pdf_details["content"] = full_result[1]
                    self.pdf_details["source"] = full_result[2]
                    self.pdf_details["page"] = full_result[3]
                output += f"\n{prompt.OBSERVATION_TOKEN} {full_result[0]}\n{prompt.THOUGHT_TOKEN}"
            else: 
                print(f"Unable to find result while using Tool {tool}")
                output += f"\n{prompt.OBSERVATION_TOKEN} {tool_result}\n{prompt.THOUGHT_TOKEN}"
            previous_responses.append(output)

    def _get_next_action(self, prompt: str) -> Tuple[str, str, str]:
        # Use the LLM to generate the Agent's next action
        result = self.chatllm.generate(prompt, stop=self.stop_pattern).content
        # Parse the result
        tool, tool_input = self._get_tool_and_input(result)
        return result, tool, tool_input

    def _get_tool_and_input(self, generated: str) -> Tuple[str, str]:
        if prompt.FINAL_ANSWER_TOKEN in generated:
            return "Final Answer", generated.split(prompt.FINAL_ANSWER_TOKEN)[-1].strip()
        regex = r"Action: [\[]?(.*?)[\]]?[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, generated, re.DOTALL)
        if not match:
            raise ValueError(f"Output of LLM is not parsable for next tool use: `{generated}` \n and match {match}")
        tool = match.group(1).strip()
        tool_input = match.group(2)
        return tool, tool_input.strip(" ").strip('"')