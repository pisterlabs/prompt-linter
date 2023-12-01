import anthropic
import openai
import os


class ChatGPTClient:
    """
    Client for the chat GPT model.
    """

    def __init__(self):
        self.model = "gpt-3.5-turbo"
        self.messages = [
            {
                "role": "system",
                "content": "You are a helpfull assistant. You do your best to assist the user in any task as an expert. You ask for clarifying questions when necessary. You try to be as consise as possible.",
            }
        ]

    def complete(self, question):
        self.messages.append({"role": "user", "content": question})
        resp = openai.ChatCompletion.create(model=self.model, messages=self.messages)
        message = resp["choices"][0]["message"]
        self.messages.append(message)
        return message["content"]


class ClaudeClient:

    """
    Client for the anthropic Claude model.

    """

    def __init__(self):
        self.client = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])
        self.prompt = f"{anthropic.HUMAN_PROMPT}"

    def complete(self, question, max_tokens_to_sample=100):
        self.prompt = f"{self.prompt} {question} {anthropic.AI_PROMPT}"

        resp = self.client.completion(
            prompt=self.prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT, anthropic.AI_PROMPT],
            model="claude-v1",
            max_tokens_to_sample=max_tokens_to_sample,
        )

        # get answer out of the response
        answer = resp["completion"]

        # update prompt
        self.prompt = f"{self.prompt} {answer}"

        return answer
