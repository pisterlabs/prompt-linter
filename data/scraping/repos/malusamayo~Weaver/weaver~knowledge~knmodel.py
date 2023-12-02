import json
from os.path import exists
from os import getenv
import os
import openai
import tiktoken

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

class LanguageModel(object):
    def __init__(self):
        pass

    def __call__(self):
        pass

# class ChatGPTModel(LanguageModel):

#     def __init__(self, config_file):
#         super().__init__()
#         self.model = Chatbot(self.config(config_file)["api_key"])

#     def config(self, config_file):
#         with open(config_file, encoding="utf-8") as f:
#             config = json.load(f)
#         return config

#     def __call__(self, prompt, rollback=True):
#         response = self.model.ask(prompt)
#         # print(response)
#         if rollback:
#             self.model.rollback(1)
#         return response["choices"][0]["text"]


ENCODER = tiktoken.get_encoding("gpt2")
def get_max_tokens(prompt: str) -> int:
    """
    Get the max tokens for a prompt
    """
    return 4000 - len(ENCODER.encode(prompt))

class GPT3Model(LanguageModel):

    def __init__(self, api_key: str = None) -> None:
        super().__init__()
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        assert openai.api_key is not None, "Please provide an OpenAI API key"

    def _get_completion(
        self,
        prompt: str,
        temperature: float = 0.5,
        max_tokens: int=256,
        n: int=1,
        stream: bool = False,
    ):
        """
        Get the completion function
        """
        return openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=["\n\n\n"],
            stream=stream,
            user="kgtest"
        )

    @retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(6))
    def __call__(self, 
                 prompt, 
                 temperature: float = 0.5,
                 max_tokens: int=256,
                 n: int=1,
                 stream: bool = False):
        response = self._get_completion(prompt,
                                        temperature=temperature,
                                        max_tokens=max_tokens,
                                        n=n,
                                        stream=stream)
        messages = [c["text"] for c in response["choices"]]
        messages = messages[0] if len(messages) == 1 else messages
        return messages
    
class GPT3ModelAsync(LanguageModel):

    def __init__(self, api_key: str = None) -> None:
        super().__init__()
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        assert openai.api_key is not None, "Please provide an OpenAI API key"

    async def _get_completion(
        self,
        prompt: str,
        temperature: float = 0.5,
        stream: bool = False,
    ):
        """
        Get the completion function
        """
        return await openai.Completion.acreate(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            temperature=temperature,
            max_tokens=256,
            stop=["\n\n\n"],
            stream=stream,
            user="kgtest"
        )

    async def __call__(self, prompt):
        response = self._get_completion(prompt)
        return await response["choices"][0]["text"]
    
class CurieModel(LanguageModel):

    def __init__(self, api_key: str = None) -> None:
        super().__init__()
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        assert openai.api_key is not None, "Please provide an OpenAI API key"

    def _get_completion(
        self,
        prompt: str,
    ):
        """
        Get the completion function
        """
        return openai.Completion.create(
            engine="curie",
            prompt=prompt,
            temperature=1.0,
            top_p=0.95,
            max_tokens=100,
            stop=["\""],
            user="kgtest"
        )

    @retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(6))
    def __call__(self, prompt):
        response = self._get_completion(prompt)
        messages = [c["text"] for c in response["choices"]]
        messages = messages[0] if len(messages) == 1 else messages
        return messages

class ChatGPTModel(LanguageModel):
    def __init__(self, sys_msg: str, api_key: str = None, temparature=1.0) -> None:
        super().__init__()
        openai.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        assert openai.api_key is not None, "Please provide an OpenAI API key"
        self.sys_msg = {"role": "system", "content": sys_msg}
        self.temperature = temparature

    @retry(wait=wait_random_exponential(min=2, max=60), stop=stop_after_attempt(6))
    def __call__(self, messages):
        messages = [self.sys_msg] + messages
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            user="kgtest",
            temperature=self.temperature,
        )
        messages = [c["message"] for c in response["choices"]]
        messages = messages[0] if len(messages) == 1 else messages
        return messages

if __name__ == "__main__":
    model = GPT3Model()
    print(model('Hi!', n=2))