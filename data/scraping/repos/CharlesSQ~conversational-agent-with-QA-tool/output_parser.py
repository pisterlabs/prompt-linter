from langchain.agents import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from langchain.output_parsers.json import parse_json_markdown


class CustomOutputParser(AgentOutputParser):

    def parse(self, text: str) -> any:
        try:
            # this will work IF the text is a valid JSON with action and action_input
            response = parse_json_markdown(text)
            action, action_input = response["action"], response["action_input"]
            if action == "Final Answer":
                # this means the agent is finished so we call AgentFinish
                return AgentFinish({"output": action_input}, text)
            else:
                # otherwise the agent wants to use an action, so we call AgentAction
                return AgentAction(action, action_input, text)
        except Exception:
            # sometimes the agent will return a string that is not a valid JSON
            # often this happens when the agent is finished
            # so we just return the text as the output
            return AgentFinish({"output": text}, text)
