from langchain.agents.conversational_chat.output_parser import ConvoOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from typing import Union
from langchain.output_parsers.json import parse_json_markdown
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS

class MyVertexOutputParser(ConvoOutputParser):
    def __init__(self):
        super().__init__()

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Attempts to parse the given text into an AgentAction or AgentFinish.

        Raises:
             OutputParserException if parsing fails.
        """
        #print ("try", text, "end try")
        try:
            # Attempt to parse the text into a structured format (assumed to be JSON
            # stored as markdown)
            #print ("before", text, "end before")
            text = text.strip()

            

            if "{" in text and "}" in text:
                text = text[text.find('{'):text.rfind('}')+1]


            #print ("after", text, "end after")

            response = parse_json_markdown(text)

            # If the response contains an 'action' and 'action_input'
            if "action" in response and "action_input" in response:
                action, action_input = response["action"], response["action_input"]

                # If the action indicates a final answer, return an AgentFinish
                if action == "Final Answer":
                    return AgentFinish({"output": action_input}, text)
                else:
                    # Otherwise, return an AgentAction with the specified action and
                    # input
                    return AgentAction(action, action_input, text)
            else:
                # If the necessary keys aren't present in the response, raise an
                # exception
                raise OutputParserException(
                    f"Missing 'action' or 'action_input' in LLM output: {text}"
                )
        except Exception as e:
            # If any other exception is raised during parsing, also raise an
            # OutputParserException
            raise OutputParserException(f"Could not parse LLM output: '{text}'") from e
        

    @property
    def _type(self) -> str:
        return "conversational_chat"