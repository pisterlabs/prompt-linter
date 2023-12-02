from openai import OpenAI
from intrat.ai_prompts.system_prompt import system_prompt

class WorkoutAI:
    def __init__(self, model: str = "gpt-3.5-turbo", temperature = 0.3, api_key: str = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.messages = [{"role": "system", "content": system_prompt}]

    def generate_response(self):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature
        )
        return response.choices[0].message.content

    def add_assistant_message(self, message):
        self.messages.append({"role": "assistant", "content": message})

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def get_response(self, user_message: str):
        self.add_user_message(user_message)
        response = self.generate_response()
        self.add_assistant_message(response)
        return response
