from typing import Any, List, Optional, Sequence, Self
from pydantic import BaseModel
from oneapi.clients.abc_client import AbstractConfig, AbstractClient
import anthropic
import os 
import json

class AnthropicConfig(AbstractConfig):
    api_key: str
    api_base: str = "https://api.anthropic.com"
    api_type: str = "anthropic"


class AnthropicDecodingArguments(BaseModel):
    prompt: str
    model: str = "claude-2.1"
    max_tokens_to_sample: int = 2048
    temperature: float = 1
    top_p: float = -1
    top_k: int = -1
    stream: bool = False
    stop_sequences: Optional[Sequence[str]] = [anthropic.HUMAN_PROMPT]


class AnthropicClient(AbstractClient):
    """
    https://github.com/anthropics/anthropic-sdk-python
    """
    def __init__(self, config : AbstractConfig) -> None:
        super().__init__(config)
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key, base_url=config.api_base)
        self.aclient = anthropic.AsyncAnthropic(api_key=config.api_key, base_url=config.api_base)
    
    @classmethod
    def from_config(cls, config: dict = None, config_file: str = "") -> Self:
        if isinstance(config_file, str) and os.path.isfile(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
        if not config:
            raise ValueError("config is empty, pass a config file or a config dict")
        return cls(AnthropicConfig(**config))
    
    def format_prompt(self, prompt: str|list[str]|list[dict], system: str = "") -> str:
        if isinstance(prompt, str):
            if not system:
                return f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}"
            else:
                return f"{anthropic.HUMAN_PROMPT} {system}\n\n{prompt}{anthropic.AI_PROMPT}"
        elif isinstance(prompt, list) and isinstance(prompt[0], str):
            msg_list = [f"{anthropic.HUMAN_PROMPT} {p}" if i%2 == 0 else f"{anthropic.AI_PROMPT} {p}" for i, p in enumerate(prompt)]
            if system:
                msg_list[0] = f"{anthropic.HUMAN_PROMPT} {system}\n\n{prompt[0]}"
            msg_list.append(anthropic.AI_PROMPT)
            return "".join(msg_list)
        elif isinstance(prompt, list) and isinstance(prompt[0], dict):
            msg_list = [f"{anthropic.HUMAN_PROMPT} {p['content']}" if p['role'] != 'assistant' else f"{anthropic.AI_PROMPT} {p['content']}" for i, p in enumerate(prompt)]
            if system:
                msg_list[0] = f"{anthropic.HUMAN_PROMPT} {system}\n\n{prompt[0]['content']}"
            msg_list.append(anthropic.AI_PROMPT)
            return "".join(msg_list)
        else:
            raise AssertionError(f"Prompt must be either a string, list of strings, or list of dicts. Got {type(prompt)} instead.")
                
    def chat_stream(self, resp):
        for data in resp:
            if data.stop_reason == 'stop_sequence':
                break
            yield data.completion

    def chat(self, prompt: str | list[str] | list[dict], system: str = "", max_tokens: int = 1024, **kwargs):
        # OpenAI use 'stop'
        if 'stop' in kwargs and kwargs['stop']:
            kwargs['stop_sequences'] = kwargs.pop('stop')
        args = AnthropicDecodingArguments(prompt=self.format_prompt(prompt=prompt, system=system), max_tokens_to_sample=max_tokens, **kwargs)
        if "verbose" in kwargs and kwargs["verbose"]:
            print(f"reqeusts args = {json.dumps(args.model_dump(), indent=4, ensure_ascii=False)}")
        resp = self.client.completions.create(**args.model_dump())
        if args.stream:
            return self.chat_stream(resp)
        else:
            return resp.completion
                
    async def achat(self, prompt: str | list[str] | list[dict], system: str = "", max_tokens: int = 1024, **kwargs):
        # OpenAI use 'stop'
        if 'stop' in kwargs and kwargs['stop']:
            kwargs['stop_sequences'] = kwargs.pop('stop')
        args = AnthropicDecodingArguments(prompt=self.format_prompt(prompt=prompt, system=system), max_tokens_to_sample=max_tokens, **kwargs)
        if "verbose" in kwargs and kwargs["verbose"]:
            print(f"reqeusts args = {json.dumps(args.model_dump(), indent=4, ensure_ascii=False)}")
        resp = await self.aclient.completions.create(**args.model_dump())
        if args.stream:
            full_comp = ""
            async for data in resp:
                if data.stop_reason == 'stop_sequence':
                    break
                full_comp += data.completion

            return full_comp
        else:
            return resp.completion

    def count_tokens(self, texts: List[str], model: str = "") -> int:
        return sum([self.client.count_tokens(text) for text in texts])