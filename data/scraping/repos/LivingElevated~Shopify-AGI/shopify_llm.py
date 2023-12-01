import os
import openai
from typing import Type, Optional, List, Dict

from pydantic import BaseModel, Field

from superagi.agent.agent_prompt_builder import AgentPromptBuilder
from superagi.helper.token_counter import TokenCounter
from superagi.tools.base_tool import BaseTool
from superagi.config.config import get_config
from superagi.llms.base_llm import BaseLlm
from superagi.resource_manager.file_manager import FileManager
from superagi.lib.logger import logger
from superagi.tools.tool_response_query_manager import ToolResponseQueryManager


class LLMInput(BaseModel):
    task_description: Optional[str] = Field(
        None,
        description="Task description."
    )
    task_function: Optional[str] = Field(
        None,
        description="Function to be performed."
    )
    product_details: Optional[Dict[str, str]] = Field(
        None,
        description="Details about the product."
    )

class ShopifyLLM(BaseTool):
    """
    Thinking Tool
    Attributes:
        name : The name.
        description : The description.
        args_schema : The args schema.
        llm: LLM used for thinking.
    """
    name = "Shopify LLM"
    description = "Shopify AI Thinking Tool"
    args_schema: Type[LLMInput] = LLMInput
    llm: Optional[BaseLlm] = None
    goals: List[str] = []
    resource_manager: Optional[FileManager] = None
    tool_response_manager: Optional[ToolResponseQueryManager] = None

    class Config:
        arbitrary_types_allowed = True

    def _execute(self, task_description: Optional[str] = None, task_function: Optional[str] = None, product_details: Optional[Dict[str, str]] = None) -> str:
        """
        Execute the Thinking tool.
        Args:
            task_description : The task description.
            task_function: Function to be performed.
            product_details: A dictionary containing product details.
        Returns:
            AI-generated reasoning for the task.
        """

        if task_description:
            prompt = task_description
        elif task_function and product_details:
            prompt_functions = {
                'generate_description': f"Write a captivating product description for {product_details.get('title', 'the product')}.",
                'generate_title': f"Generate a catchy product title for the following product: {product_details.get('description', '')}",
                'generate_tags': f"Generate some relevant tags for the following product: {product_details.get('description', '')}",
                'generate_vendor': f"Generate a suitable vendor for the following product: {product_details.get('description', '')}",
                'generate_price': f"Generate a suitable price for the following product: {product_details.get('description', '')}",
                'generate_product_type': f"Generate a suitable product type for the following product: {product_details.get('description', '')}"
            }
            prompt = prompt_functions.get(task_function, f"Perform the task.")
        else:
            raise ValueError(
                "Either 'task_description' or both 'task_function' and 'product_details' must be provided.")

        messages = [{"role": "system", "content": prompt}]
        result = self.llm.chat_completion(
            messages,
            max_tokens=self.max_token_limit
        )

        return result["content"]