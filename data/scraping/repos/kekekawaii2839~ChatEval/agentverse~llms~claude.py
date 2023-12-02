import logging
import os

from pydantic import BaseModel, Field

from agentverse.llms.base import LLMResult

from . import llm_registry
from .base import BaseChatModel, BaseCompletionModel, BaseModelArgs

logger = logging.getLogger(__name__)

try:
    from anthropic import Anthropic, AsyncAnthropic, HUMAN_PROMPT, AI_PROMPT
except ImportError:
    is_anthropic_available = False
    logging.warning("anthropic package is not installed")
else:
    Anthropic.api_key = os.environ.get("ANTHROPIC_API_KEY")
    AsyncAnthropic.api_key = os.environ.get("ANTHROPIC_API_KEY")
    if Anthropic.api_key is None:
        logging.warning(
            "Anthropic API key is not set. Please set the environment variable ANTHROPIC_API_KEY"
        )
        is_anthropic_available = False
    else:
        is_anthropic_available = True


class ClaudeChatArgs(BaseModelArgs):
    model: str = "claude-2"
    max_tokens_to_sample: int = 300
    temperature: float = Field(default=1.0)

@llm_registry.register("claude-2")
class ClaudeChat(BaseChatModel):
    args: ClaudeChatArgs = Field(default_factory=ClaudeChatArgs)

    def __init__(self, max_retry: int = 3, **kwargs):
        args = ClaudeChatArgs()
        args = args.dict()

        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        if len(kwargs) > 0:
            logging.warning(f"Unused arguments: {kwargs}")
        super().__init__(args=args, max_retry=max_retry)
    
    def _construct_messages(self, prompt: str):
        return f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}"
    
    def generate_response(self, prompt: str) -> LLMResult:
        messages = self._construct_messages(prompt)
        anthropic = Anthropic(proxies={"http://": "http://127.0.0.1:33210", "https://": "http://127.0.0.1:33210"})
        result = anthropic.completions.create(prompt=messages, **self.args.dict())
        return LLMResult(
            content=result.completion,
            send_tokens=0,
            recv_tokens=0,
            total_tokens=0
        )
    
    async def agenerate_response(self, prompt: str) -> LLMResult:
        messages = self._construct_messages(prompt)
        anthropic = AsyncAnthropic(proxies={"http://": "http://127.0.0.1:33210", "https://": "http://127.0.0.1:33210"})
        result = await anthropic.completions.create(prompt=messages, **self.args.dict())
        return LLMResult(
            content=result.completion,
            send_tokens=0,
            recv_tokens=0,
            total_tokens=0
        )