from langchain import OpenAI, LLMChain
from langchain.schema import AgentAction, AgentFinish
import re
import os
from langchain.agents import initialize_agent, Tool, ZeroShotAgent, AgentExecutor
from typing import Any, List, Optional, Tuple, Union

class QAAgent(ZeroShotAgent):
  """Agent for the MRKL chain."""
  lastDocsViewed: List = None
  FINAL_ANSWER_ACTION = "Final Answer: "
  def _extract_tool_and_input(self, llm_output: str) -> Optional[Tuple[str, str]]:
      llm_output = llm_output.strip()
      if self.FINAL_ANSWER_ACTION in llm_output:
          return "Final Answer", llm_output.split(self.FINAL_ANSWER_ACTION)[-1]

      regex_action = r"Action: (.*?)\n"
      match_action = re.search(regex_action, llm_output)
      action = match_action.group(1)
      if not match_action:
          raise ValueError(f"Could not parse match_action LLM output: `{llm_output}`")
      regex_action_input = r"Action Input: (.*)"
      match_action_input = re.search(regex_action_input, llm_output)
      action_input = match_action_input.group(1)
      if not match_action_input:
          raise ValueError(f"Could not parse match_action_input LLM output: `{llm_output}`")
      return action, action_input.strip(" ").strip('"')

  def doc_extractor(self, observation):
    # assume it's a tuple that can take in 2 inputs -> (the doc results,the doc links)
    self.lastDocsViewed = observation[1]
    return observation[0]
    

  def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any) -> Union[AgentAction, AgentFinish]:
      """Given input, decided what to do.

      Args:
          intermediate_steps: Steps the LLM has taken to date,
              along with observations
          **kwargs: User inputs.

      Returns:
          Action specifying what tool to use.
      """
      thoughts = ""
      for action, observation in intermediate_steps:
          if type(observation) == tuple:
            observation = self.doc_extractor(observation)
          thoughts += action.log
          thoughts += f"\n{self.observation_prefix}{observation}\n{self.llm_prefix}"
      new_inputs = {"agent_scratchpad": thoughts, "stop": "Observation:"}
      full_inputs = {**kwargs, **new_inputs}
      full_output = self.llm_chain.predict(**full_inputs)
      parsed_output = self._extract_tool_and_input(full_output)
      predict_output = None
      while parsed_output is None:
          full_output = self._fix_text(full_output)
          full_inputs["agent_scratchpad"] += full_output
          output = self.llm_chain.predict(**full_inputs)
          predict_output = output
          full_output += output
          parsed_output = self._extract_tool_and_input(full_output)
      tool, tool_input = parsed_output
      if tool == self.finish_tool_name:
          # self.lastDocsViewed = reference_link_builder(self.lastDocsViewed, "https://github.com/hwchase17/langchain")
          tool_input = {"response": tool_input, "references": "\n Here is a list of documents that I viewed: " + str(self.lastDocsViewed)}
          return AgentFinish({"output": tool_input}, full_output)
      output = AgentAction(tool, tool_input, full_output)
      return output
