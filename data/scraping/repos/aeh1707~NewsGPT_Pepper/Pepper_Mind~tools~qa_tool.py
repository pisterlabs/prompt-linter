from typing import Any, List, Optional, Union, cast
from steamship import Steamship
import random
from steamship import Block, Steamship, Tag, Task
from steamship.agents.llms import OpenAI
from steamship.agents.schema import AgentContext, Tool
from steamship.agents.utils import get_llm, with_llm
from steamship.data.plugin.index_plugin_instance import EmbeddingIndexPluginInstance
from steamship.utils.repl import ToolREPL
from steamship.agents.tools.question_answering import VectorSearchQATool
from steamship.agents.utils import get_llm, with_llm
import uuid
import requests

class StudentAdvisorTool(Tool):
    name: str = "StudentAdvisorTool"
    human_description: str = "a simple qa tool to help students only about university related questions."
    agent_description: str = "a simple qa tool to help students only about university related questions."

    def answer(self, question: str, context: AgentContext) -> List[Block]:
        url = "https://abdelhadi-hireche.steamship.run/qa-bot-telegram-j21/qa-bot-telegram-j21/answer"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer 26721291-D416-4C30-AA6A-304FC4E0BC5F"
        }
        data = {
            "question": question
        }

        response = requests.post(url, headers=headers, json=data)

        answer = response.json()['answer']

        return [Block(text=answer)]

    def run(self, tool_input: List[Block], context: AgentContext) -> Union[List[Block], Task[Any]]:
        output = []
        for input_block in tool_input:
            if not input_block.is_text():
                continue
            for output_block in self.answer(input_block.text, context):
                output.append(output_block)
        return output
