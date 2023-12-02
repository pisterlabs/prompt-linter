from typing import Optional, Tuple

from langchain.agents import AgentExecutor
from langchain.schema import AgentFinish, AgentAction

from yid_langchain_extensions.output_parser.action_parser import ActionWithThoughts


class AgentWithThoughtsExecutor(AgentExecutor):
    def _get_tool_return(
        self, next_step_output: Tuple[AgentAction, str]
    ) -> Optional[AgentFinish]:
        agent_action, observation = next_step_output
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        if agent_action.tool in name_to_tool_map and name_to_tool_map[agent_action.tool].return_direct:
            return_values = {self.agent.return_values[0]: observation}
            if isinstance(agent_action, ActionWithThoughts):
                return_values.update(agent_action.all_thoughts)
            return AgentFinish(
                return_values,
                agent_action.log
            )
        return None
