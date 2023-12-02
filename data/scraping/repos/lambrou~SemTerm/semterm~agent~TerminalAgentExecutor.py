from abc import ABC
from typing import Dict, List, Tuple, Union, Optional

from langchain.agents import AgentExecutor
from langchain.agents.tools import InvalidTool
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool

from semterm.langchain_extensions.schema import AgentMistake


class TerminalAgentExecutor(AgentExecutor, ABC):
    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Union[
        AgentFinish, List[Tuple[AgentAction, str]], List[Tuple[AgentMistake, str]]
    ]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        # Call the LLM to see what to do.
        output = self.agent.plan(intermediate_steps, **inputs)
        result = []
        actions: List[AgentAction]
        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            return output
        if isinstance(output, (AgentAction, AgentMistake)):
            actions = [output]
        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(  # pragma: no cover
                    agent_action,
                    verbose=self.verbose,
                    color="green",
                )
            # Otherwise we lookup the tool
            if agent_action.tool in name_to_tool_map:
                tool = name_to_tool_map[agent_action.tool]
                return_direct = tool.return_direct
                color = color_mapping[agent_action.tool]
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                if return_direct:
                    tool_run_kwargs["llm_prefix"] = ""
                # We then call the tool on the tool input to get an observation
                observation = tool.run(
                    agent_action.tool_input,
                    verbose=self.verbose,
                    color=color,
                    **tool_run_kwargs,
                )
            else:
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = InvalidTool().run(
                    agent_action.tool,
                    verbose=self.verbose,
                    color=None,
                    **tool_run_kwargs,
                )
            result.append((agent_action, observation))
        return result
