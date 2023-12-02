from abc import ABC, abstractmethod

import openai
import anthropic
from tenacity import retry, stop_after_attempt, wait_random_exponential


class BaseComplete(ABC):
    @abstractmethod
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
    def complete(self, content: str, **kwargs) -> str:
        """
        Generate text from the given content.
        Args:
            content: raw text input that all models share
            **kwargs: model-specific arguments

        Returns: generated text
        """
        pass


class OpenaiComplete(BaseComplete):
    # This code is ran with openai==0.27.10. The API may have changed since then.
    def __init__(self, api_key_path: str):
        with open(api_key_path, "r") as f:
            openai.api_key = f.read()

    def complete(self, content: str, **kwargs) -> str:
        messages = [{"role": "user", "content": content}]
        response = openai.ChatCompletion.create(messages=messages, **kwargs)
        return response.choices[0]['message']['content']


class AnthropicComplete(BaseComplete):
    # This code is ran with anthropic==0.2.10. The API may have changed since then.
    def __init__(self, api_key_path: str):
        with open(api_key_path, "r") as f:
            self.api_key = f.read()
        self.client = anthropic.Client(self.api_key)

    def complete(self, content: str, **kwargs) -> str:
        prompt = f"{anthropic.HUMAN_PROMPT} {content}{anthropic.AI_PROMPT}"
        response = self.client.completion(prompt=prompt, stop_sequences=[anthropic.HUMAN_PROMPT], **kwargs)
        return response['completion'][1:]  # first char is expected to be a space


# test
if __name__ == '__main__':
    import yaml

    with open("../config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    content = "What was Alan Turing's favorite McDonald menu?"

    # openai
    openai_complete = OpenaiComplete(config['openai_key_path'])
    print(openai_complete.complete(content, model="gpt-3.5-turbo", max_tokens=100))

    # anthropic
    anthropic_complete = AnthropicComplete(config['anthropic_key_path'])
    print(anthropic_complete.complete(content, model="claude-1", max_tokens_to_sample=100))
