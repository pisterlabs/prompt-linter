import os
import openai
from superagi.tools.base_tool import BaseTool
from pydantic import BaseModel, Field
from typing import Type

# Defining the Input Model
class GorillaToolInput(BaseModel):
    message: str = Field(..., description="Message to be processed by Gorilla LLM")

# Defining the Gorilla LLM Tool
class GorillaTool(BaseTool):
    name: str = "Gorilla LLM Tool"
    args_schema: Type[BaseModel] = GorillaToolInput
    description: str = "Tool to interact with Gorilla LLM"
    
    def _execute(self, message: str = None):
        try:
            response = self.get_gorilla_response(message)
            return response
        except Exception as e:
            return f"Error: {str(e)}"

    def get_gorilla_response(self, prompt):
        openai.api_base = os.environ.get('GORILLA_LLM_API_ENDPOINT', 'http://zanino.millennium.berkeley.edu:8000/v1')

        model = "gorilla-7b-hf-v1"  # You can adjust this if needed
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"Gorilla LLM Error: {str(e)}")
