from langchain.tools import BaseTool
from typing import Any
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class PlannerTool(BaseTool):
    name = "Planner tool"
    description = "Use this tool when you have complex task and need to generate a list of subtasks for a high level task"
    model = "gpt-4"

    def read_prompts(self):
        examples = '../prompts/planner.txt'
        with open(examples, 'r') as file:
            prompts = file.read()
        return prompts

    def _run(self, task: str):
        prompts = self.read_prompts()

        initial_message = {
            "role": "system",
            "content": "You are planning high level tasks that will be executed. Be explicit and concise. Use the lowest possible number of subtasks to accomplish the task."
        }

        query_message = {
            "role": "user",
            "content": f"Given the task '{task}', execute '{prompts}'."
        }

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[initial_message, query_message],
            temperature=0.1,  # Lower temperature to make output more deterministic
            max_tokens=100  # Limit output to 100 tokens
        )

        task_breakdown = response['choices'][0]['message']['content']

        # Re-evaluate and simplify task list
        simplification_message = {
            "role": "user",
            "content": f"Given the subtask breakdown '{task_breakdown}', format the subtasks as a Python list of values."
        }

        simplification_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[initial_message, simplification_message],
            temperature=0.5,  # Lower temperature to make output more deterministic
            max_tokens=100  # Limit output to 100 tokens
        )

        simplified_task_breakdown = simplification_response['choices'][0]['message']['content']

        return simplified_task_breakdown
    

    def _arun(self, task: Any):
        raise NotImplementedError("This tool does not support async")
