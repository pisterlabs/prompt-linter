from openai import OpenAI
import os


MODEL = "gpt-3.5-turbo"
API_KEY = os.getenv("OPENAI_API_KEY")


class ChatCompletion:
    def __init__(self) -> None:
        self.model = MODEL
        self.temperature = 0
        self.max_tokens = 0

    def create(self, messages: list = []):
        client = OpenAI(api_key=API_KEY)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return response


def main():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Knock knock."},
        {"role": "assistant", "content": "Who's there?"},
        {"role": "user", "content": "Orange."},
    ]
    chat = ChatCompletion()
    print(chat.create(messages=messages))


if __name__ == "__main__":
    main()
