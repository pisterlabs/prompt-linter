from typing import List
from openai import OpenAI
from openai.types.beta.assistant_create_params import Tool

client = OpenAI().beta

class Assistant:
    id: str

    def __init__(
        self, 
        tools: List[Tool], 
        name: str = "Assistant", 
        instructions: str = "You are a helpful assistant.", 
        model: str = "gpt-3.5-turbo-1106"
    ):
        self._assistant = client.assistants.create(
            name=name,
            instructions=instructions,
            tools=tools,
            model=model,
        )
        self.id = self._assistant.id
        
    def get_by_name(name: str) -> str:
        for assistant in client.assistants.list().data:
            if assistant.name == name:
                return assistant.id
            
