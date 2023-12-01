from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from constants import SYSTEM_PROMPT_TEMPLATE, SYSTEM_PROMPT_CORRECTIONS, SYSTEM_PROMPT_EXPLANATION

class TandamGPT:
    def __init__(self):
        # Add your open ai API key
        self.llm = ChatOpenAI(openai_api_key="", model="gpt-3.5-turbo", verbose=True)
    
    def set_language(self, language: str):
        system_prompt = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE.format(language=language))
        self.messages = [system_prompt]

    # Add a message to the message history based on the specified role ('user' or 'Ai').
    def add_to_history(self, message, role):
        if role == "user":
            new_msg = HumanMessage(content=message)
        elif role == "Ai":
            new_msg = AIMessage(content=message)
        else:
            raise ValueError("Invalid role: role must be either 'user' or 'Ai'.")

        self.messages.append(new_msg)
    
    # Get the response using the ChatOpenAI instance with the given messages.
    def get_response(self, messages):
        return self.llm(messages=messages).content

    def receive(self, text: str):
        # Add the user's input to the message history.
        self.add_to_history(text, "user")
        # Generate AI response based on the current message history.
        response = self.get_response(self.messages)
        # Add AI response to the message history.
        self.add_to_history(response, "Ai")

        return response
    
    # Generate the corrections of the user's sentences
    def correction(self, message):
        prompt = [SystemMessage(content=SYSTEM_PROMPT_CORRECTIONS)]
        prompt.append(HumanMessage(content=message))
        response = self.get_response(prompt)

        return response
    
    # Generate the explanantion of the AI's response
    def explanation(self, message):
        prompt = [SystemMessage(content=SYSTEM_PROMPT_EXPLANATION)]
        prompt.append(HumanMessage(content=message))
        response = self.get_response(prompt)

        return response
