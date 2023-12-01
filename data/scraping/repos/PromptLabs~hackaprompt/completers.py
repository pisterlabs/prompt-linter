import logging
import os
from abc import ABC, abstractmethod

import openai
import tiktoken
from gradio_client import Client
from transformers import T5Tokenizer

HUB_TOKEN = os.getenv("HUB_TOKEN")
FLAN_HF_SPACE = "jerpint/i-like-flan"
LLAMA_HF_SPACE = "meta-llama/Llama-2-7b"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextCompletion(ABC):
    @abstractmethod
    def get_token_count(prompt: str) -> int:
        ...

    @abstractmethod
    def complete(prompt: str):
        ...

    def get_score_multiplier(self) -> float:
        return 1.0


class DebugCompleter(TextCompletion):
    """This is only meant for testing/debugging purposes."""

    def __init__(self, model):
        self.model = model

    def get_token_count(self, prompt: str) -> int:
        encoding = tiktoken.encoding_for_model("text-davinci-003")
        return len(encoding.encode(prompt))

    def complete(self, prompt: str):
        if "$CORRECT$" in prompt:
            # returns the correct prompt
            return "I have been PWNED"
        elif "$ERROR$" in prompt:
            # simulate an error
            raise RuntimeError(f"User raised an error explicitly in {self.model}")
        else:
            return "Some incorrect completion"


class GPTCompletor(TextCompletion):
    def __init__(self, openai_api_key, model):
        self.openai_api_key = openai_api_key
        self.model = model

    def get_token_count(self, prompt: str) -> int:
        encoding = tiktoken.encoding_for_model("text-davinci-003")
        return len(encoding.encode(prompt))

    def complete(self, prompt: str):
        # Set the api key
        openai.api_key = self.openai_api_key

        # Call the API to generate a response
        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            max_tokens=1024,
            temperature=0,
            top_p=0,
            frequency_penalty=1,
            presence_penalty=1,
        )

        # Get the response text
        response_text = response["choices"][0]["text"]
        return response_text


class ChatGPTCompletor(TextCompletion):
    def __init__(self, openai_api_key, model):
        self.openai_api_key = openai_api_key
        self.model = model

    def get_token_count(self, prompt: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(prompt))

    def complete(self, prompt: str):
        # Set the api key
        openai.api_key = self.openai_api_key

        messages = [
            {"role": "user", "content": prompt},
        ]

        # Call the API to generate a response
        response = openai.ChatCompletion.create(
            messages=messages,
            model=self.model,
            temperature=0,
            request_timeout=20,
        )

        # Get the response text
        response_text = response["choices"][0]["message"]["content"]

        return response_text

    def get_score_multiplier(self) -> float:
        return 2.0


class FlanT5Completor(TextCompletion):
    def __init__(self, model: str):

        self.model = model
        self.client = Client(src=FLAN_HF_SPACE, hf_token=HUB_TOKEN)

    def get_token_count(self, prompt: str) -> int:
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        token_count = len(input_ids[0])
        return token_count

    def complete(self, prompt: str) -> str:
        response_text = self.client.predict(prompt)
        return response_text


class LLama2Completor(TextCompletion):
    def __init__(self, model: str):

        self.model = model
        # self.client = Client(src=LLAMA_HF_SPACE, hf_token=HUB_TOKEN)
        self.client = Client(src="https://e554306b558a76b0fd.gradio.live")


    def get_token_count(self, prompt: str) -> int:
        return 0

    def complete(self, prompt: str) -> str:
        output = self.client.predict(prompt)
        return output




class ClaudeCompletor(TextCompletion):
    def __init__(self, model="claude-2"):
        from anthropic import Anthropic
        import os

        api_key=os.environ["ANTHROPIC_API_KEY"]

        self.model = model
        self.anthropic = Anthropic(api_key=api_key)

    def get_token_count(self, prompt: str):
        "Not used in actual competition, so just dummy value"
        return 0

    def complete(self, prompt: str):

        from anthropic import HUMAN_PROMPT, AI_PROMPT

        completion = self.anthropic.completions.create(
            model=self.model,
            max_tokens_to_sample=300,
            prompt=f"{HUMAN_PROMPT} {prompt} {AI_PROMPT}",


        )
        return completion.completion


completers = {
    "text-davinci-003": GPTCompletor,
    "FlanT5-XXL": FlanT5Completor,
    "gpt-3.5-turbo": ChatGPTCompletor,
    "gpt-4-0613": ChatGPTCompletor,
    "gpt-3.5-turbo-0613": ChatGPTCompletor,
    "claude-2": ClaudeCompletor,
    "llama-2": LLama2Completor,
}


def get_completer(model: str, openai_api_key: str = ""):
    logger.info(f"Loading completer for {model=}")
    if model in ["text-davinci-003", "gpt-3.5-turbo", "gpt-4-0613", "gpt-3.5-turbo-0613"]:
        completer = completers[model](model=model, openai_api_key=openai_api_key)
    elif model in ["FlanT5-XXL", "claude-2", "llama-2"]:
        completer = completers[model](model=model)
    elif model == "gpt-debug":
        # Not included in completers because we dont want it showing in the app
        logger.warning("Using debugging completer...")
        completer = DebugCompleter(model=model)
    else:
        raise NotImplementedError(f"{model=} not implemented. Model must be onf of {list(completers.keys())}")

    return completer
