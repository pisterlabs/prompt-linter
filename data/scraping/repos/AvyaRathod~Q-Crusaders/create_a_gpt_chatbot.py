from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ChatMessageHistory
from langchain.callbacks import get_openai_callback

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

def ChatbotSummary(summary):
    Prompt = "You are given an Employee Feedback and are given the task to summarize what the company should improve on in short bullet points. The Feedback is the following: "
    chatbot = ChatBot(openai_api_key="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", model="gpt-3.5-turbo")
    return chatbot.start_chat(Prompt + summary)

"""
Example usage:
Fac
chatbot = ChatBot(openai_api_key="sk...", model=""gpt-3.5-turbo)
output = chatbot.start_chat("Hi")
"""
