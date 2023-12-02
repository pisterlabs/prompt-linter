import json
from langchain.agents import AgentOutputParser
# from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish
from Console.client.chains.prompts.airoboros import JSON_FUNC_INSTRUCTIONS

def parse_json(text: str) -> dict:
    """Parse the JSON output from the LLM.

    Args:
        text: JSON string.

    Returns:
        Dictionary with action and action_input.
    """
    return json.loads(text)


FORMAT_INSTRUCTIONS = JSON_FUNC_INSTRUCTIONS


class JsonOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            
            _text = text.split("<|im_end|>")[0].split("<|im_start|>")[0]
            # this will work IF the text is a valid JSON with action and action_input
            response = parse_json(_text)
            action, action_input = response["action"], response["action_input"]
            if action == "final_answer":
                # this means the agent is finished so we call AgentFinish
                return AgentFinish({"output": action_input}, _text)
            else:
                # otherwise the agent wants to use an action, so we call AgentAction
                return AgentAction(action, action_input, _text)
        except Exception as e:
            # sometimes the agent will return a string that is not a valid JSON
            # often this happens when the agent is finished
            # so we just return the text as the output
            return AgentFinish({"output": _text}, str(e) + "\n" + _text)

    @property
    def _type(self) -> str:
        return "conversational_chat"