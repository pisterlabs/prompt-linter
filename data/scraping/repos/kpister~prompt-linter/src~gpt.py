"""
Docstring

Home of the GPT Class

each instance of the GPT class is a conversation thread

Additionally, use of just "python gpt.py" is a quite nice CLI.
"""
from openai import OpenAI
import json
from typing import Generator

from dotenv import load_dotenv
import os



class GPT():
    gpt_models = ("gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview")

    def __init__(
            self,
            model: str = gpt_models[2],
            temperature: float = 0.7,
            system_message: str = "You are a helpful assistant"
        ):
        """Docstring"""
        self._model = model
        self.temperature = temperature
        self.system_message = [
            {"role": "system", "content": system_message}
        ]
        self.messages = []

        load_dotenv()
        self.client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
        
        
    @staticmethod
    def zero_shot(query: str, mode = 0, model = gpt_models[0], temperature = .7):
        gpt_instance = GPT(model, temperature)
        
        if mode == 0:
            answer = gpt_instance._conversation(query)
        
        return answer["choices"][0]["message"]["content"]

    def run(self, query: str, mode: int = 0) -> str:
        """Input to get access to all of the various response types within the GPT model
        mode 0: typical access to the GPT model aka conversational
        
        Returns a string."""
        
        if mode == 0:
            response = self._conversation(query)
            
        return response["choices"][0]["message"]["content"]
    
    def srun(self, query: str, mode: int = 0) -> Generator:
        """Essentially the same as the self.run method, but streams responses.
        
        Returns a Generator. Proper use: `for i in self.run_stream(input): print(i, end='')`"""

        self._add_message(role="user", content=query)


        response = self.client.chat.completions.create(
            model = self._model,
            temperature = self.temperature,
            messages = self.system_message + self.messages,
            stream = True
        )

        total_response = ''
        for chunk in response:
            if chunk.choices[0].delta.content is not None:

                yield chunk.choices[0].delta.content # * needs work

                total_response += chunk.choices[0].delta.content

        self._add_message(role='assistant', content=total_response)
            
    
    def _add_message(self, role: str, content: str):
        """Adds a new message to self.messages"""
        self.messages.append({"role": role, "content": content})

    def clear_messages(self):
        """Clears out old messages"""
        self.messages = []


    def update_model(self, new_model: str):
        """Updates the model that your GPT class instance is currently running on"""
        self._model = new_model

    def __str__(self):
        return f"Model: {self._model} | Temperature: {self.temperature}, Message Length: {len(self.messages)}"

    def _call_openai_api(self):
        """Docstring"""

        response = self.client.chat.completion.create(
            model = self._model,
            temperature = self.temperature,
            messages = self.system_message + self.messages
        )

        return response
    
    def _conversation(self, query: str):
        """Basic Running of AI system, with simple memory (appends new messages) returns information 
        in the form response["choices"][0]["message"]["content"]"""
        
        self._add_message(role="user", content=query)

        response = self._call_openai_api()

        self._add_message(
            role = response["choices"][0]["message"]["role"],
            content = response["choices"][0]["message"]["content"]
        )
    
        return response
    
    def set_system_message(self, message: str):
        """reads in a json file and will set system message as whatever your string corresponds"""
        filepath = "./system_message.json"
        with open(filepath, "r") as json_file:
            system_messages = json.load(json_file)
        self.system_message = [{"role": "system", "content": system_messages[message]}]

if __name__ == "__main__":
    
    # --- CLI ---
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.live import Live

    console = Console()

    ai = GPT(
        model = "gpt-4",
        # model = "gpt-3.5-turbo",
        system_message = """You are a capable and helpful assistant. If you don't know the answer to a question, tell the user. Format responses in markdown."""
    )

    def generate_panel(message: Markdown) -> Panel:
        return Panel(
            message,
            box=box.HEAVY,
            title=ai._model,
            border_style="red",
            # padding=1,
            expand=True,
            highlight=True
        )

    user_input = console.input(f"Chatting with {ai._model} | (q to quit):\nUser: ")
    # --- Conversation Loop ---
    while user_input != "q" or user_input != "Q":
        whole_message = ''

        with Live(generate_panel(message=whole_message), refresh_per_second=4) as live:

            for token in ai.srun(user_input):

                whole_message += token
                live.update(generate_panel(Markdown(whole_message)))

        print()
        user_input = console.input("User: ")
        if user_input == "c" or user_input == "C":
            ai.clear_messages()
