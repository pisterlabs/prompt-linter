import os
from typing import Any, Optional

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import Extra

import registry
import streaming
from .base import BaseTool, BASE_TOOL_DESCRIPTION_TEMPLATE

current_dir = os.path.dirname(__file__)
project_root = os.path.join(current_dir, '../')
usage_guide_path = os.path.join(project_root, 'usage_guide.md')

with open(usage_guide_path, 'r') as f:
    USAGE_GUIDE = f.read()

TEMPLATE = f'''You are an expert Web3 assistant called Cacti. You help users interact with Web3 ecosystem, such as with DeFi, NFTs, ENS, etc., by analyzing their query and providing an appropriate action in your response.
# INSTRUCTIONS
- You have access to the Markdown-formatted usage guide for this chat app below which contains some example prompts to assist users in using the app.
- Always use the usage guide to answer the user's question about the app and provide the example prompts from the guide for the suggested actions
- Do not make up any information or prompts, only use those provided in the usage guide.
- Always include the link to the full usage guide in your final response - https://github.com/yieldprotocol/cacti-backend/blob/master/usage_guide.md
- The final response should be in markdown format.

# USAGE GUIDE
{USAGE_GUIDE}

---
User: {{question}}
Assistant:'''


@registry.register_class
class AppUsageGuideTool(BaseTool):
    _chain: LLMChain

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.allow

    def __init__(
            self,
            *args,
            **kwargs
    ) -> None:
        prompt = PromptTemplate(
            input_variables=["question"],
            template=TEMPLATE,
        )
        new_token_handler = kwargs.get('new_token_handler')
        chain = streaming.get_streaming_chain(prompt, new_token_handler)

        description=BASE_TOOL_DESCRIPTION_TEMPLATE.format(
                tool_description="answer questions about the chat assistant app, what it can do, how to interact with it",
                input_description="a standalone query with all relevant contextual details pertaining to the chat web application",
                output_description="an answer to the question, with suggested follow-up questions if available",
            )
        super().__init__(
            *args,
            _chain=chain,
            description=description,
            **kwargs
        )

    def _run(self, query: str) -> str:
        example = {
            "question": query,
            "stop": "User",
        }
        result = self._chain.run(example)
        return result.strip()

    async def _arun(self, query: str) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} does not support async")