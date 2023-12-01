import pytest
from src.template.template import CustomPromptTemplate, read_template
from langchain.schema import HumanMessage, AgentAction
from langchain.agents import Tool
from langchain.prompts import BaseChatPromptTemplate
from typing import List, Dict, Any

def dummy_function():
    pass

class CustomPromptTemplateExample(BaseChatPromptTemplate):
    tools: List[Tool]
    input_variables: Dict[str, str] = {"agent_scratchpad": "scratchpad"}

    class Config:
        extra = "allow"

    def __init__(self, **data):
        super().__init__(**data)
        self.template = data.get('template')
        self.tools = data.get('tools')

    def format_messages(self, **kwargs: Any) -> List[HumanMessage]:
        tool_descriptions = "\n".join(f"{tool.name}: {tool.description}" for tool in self.tools)
        tool_names = ", ".join(tool.name for tool in self.tools)
        formatted_template = self.template.format(agent_scratchpad=kwargs["agent_scratchpad"],
                                                tool_names=tool_names,
                                                tools=tool_descriptions)
        return [HumanMessage(content=formatted_template)]



def read_template(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()

# Test cases for CustomPromptTemplate
test_cases = [
    (
        {
            "template": "Thoughts:\n{agent_scratchpad}\nAvailable Tools: {tool_names}\n\n{tools}",
            "agent_scratchpad": "Action: Tool1\nAction Input: input1\nObservation: Observation1\nThought: " \
                                "Action: Tool2\nAction Input: input2\nObservation: Observation2\nThought: ",
            "tool_names": "Tool1, Tool2",
            "tools": [
                Tool(name="Tool1", description="Description for Tool1", func=dummy_function),
                Tool(name="Tool2", description="Description for Tool2", func=dummy_function),
            ],
        },
        [
            HumanMessage(
                content="Thoughts:\nAction: Tool1\nAction Input: input1\nObservation: Observation1\nThought: "
                        "Action: Tool2\nAction Input: input2\nObservation: Observation2\nThought: "
                        "\nAvailable Tools: Tool1, Tool2\n\nTool1: Description for Tool1\nTool2: Description for Tool2"
            )
        ],
    ),
]

@pytest.mark.parametrize("kwargs, expected_result", test_cases)
def test_custom_prompt_template_format_messages(kwargs, expected_result):
    prompt_template = CustomPromptTemplateExample(template=kwargs["template"], tools=kwargs["tools"])
    result = prompt_template.format_messages(**kwargs)
    assert result == expected_result, f"Expected {expected_result}, but got {result}"

def test_read_template():
    file_path = "test_template.txt"
    expected_content = "This is a test template."

    with open(file_path, "w") as f:
        f.write(expected_content)

    content = read_template(file_path)
    assert content == expected_content, f"Expected {expected_content}, but got {content}"
