from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.callbacks import get_openai_callback
from os import environ
from dotenv import load_dotenv

load_dotenv()


class ChatBot:
    def __init__(self, openai_api_key, model):
        self.prompt = """
        write your prompt here-
        """
        self.history = ChatMessageHistory()
        self.chat = ChatOpenAI(openai_api_key=openai_api_key, model=model)
        self.history.add_user_message(self.prompt)

    def start_chat(self, user_input):
        self.history.add_user_message(user_input)

        with get_openai_callback() as cb:
            output = self.chat(self.history.messages).content
            print(output)
            print(cb)
        self.history.add_ai_message(output)

        return output


def ChatbotSummary(summary, diseaseName="Yellow Rust", cropName="Wheat"):
    Prompt = f"Give me solution to {diseaseName} for {cropName}. Make the output less than 250 words, in terms of points, and consise. Give the explanation such that a layman could understand."
    key = environ["API_KEY"]
    chatbot = ChatBot(key, "gpt-3.5-turbo")
    return chatbot.start_chat(Prompt + summary)

