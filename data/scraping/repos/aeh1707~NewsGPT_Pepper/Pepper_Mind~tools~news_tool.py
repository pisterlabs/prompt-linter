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

import json
import urllib.request
from urllib.parse import quote

apikey = "API_KEY"



class NewsTool(Tool):
    name: str = "NewsTool"
    human_description: str = "a tool that reports latest news."
    agent_description: str = "a tool that reports latest news."

    def query_news(self, question: str, context: AgentContext) -> List[Block]:
        question = quote(question)
        url = f"https://gnews.io/api/v4/search?q={question}&lang=en&country=ae&max=10&apikey={apikey}"
        articles_titles = []
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
            articles = data["articles"]
            for article in articles:
                title = article["title"]
                articles_titles.append(Block(text=title))
        return articles_titles

    def run(self, tool_input: List[Block], context: AgentContext) -> Union[List[Block], Task[Any]]:
        output = []
        for input_block in tool_input:
            if not input_block.is_text():
                continue
            for output_block in self.query_news(input_block.text, context):
                output.append(output_block)
        return output
