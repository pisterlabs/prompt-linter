from markingpal.env import set_env
from langchain.chat_models import ChatOpenAI


class FeedbackAssistant:
    def __init__(self) -> None:
        # Load the environment variables (e.g. OpenAI API key)
        set_env()

        # Load the model
        self.llm = ChatOpenAI()

    def rewrite(self, prompt):
        response = self.llm.invoke(prompt)
        return response
