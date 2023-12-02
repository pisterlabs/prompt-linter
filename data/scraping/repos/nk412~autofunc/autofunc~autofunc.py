import json
import openai
from pydantic import BaseModel

class AutoFunc():
    def __init__(self, system_instruction: str,
                 response_model: BaseModel = None,
                 model: str = "gpt-4-0613",
                 temperature: int = 0
                ):
        self.system_instruction = system_instruction
        self.model = model
        self.temperature = temperature
        self.response_model = response_model
        self.schema = response_model.model_json_schema() if response_model else None

    def __call__(self, input):
        params = {
            'model': self.model,
            'temperature': self.temperature,
            'messages': [
                {"role": "system", "content": self.system_instruction},
                {"role": "user", "content": input},
            ],
        }

        # Use function calling only if a schema is provided
        if self.schema:
            params |= {
                'functions': [
                    {
                      "name": "fn",
                      "description": "GPT function",
                      "parameters": self.schema
                    }
                ],
                'function_call': {"name": "fn"}
            }

        response = openai.ChatCompletion.create(**params)
        return response.choices[0]["message"]["content"] \
            if self.schema is None \
            else json.loads(response.choices[0]["message"]["function_call"]["arguments"])
