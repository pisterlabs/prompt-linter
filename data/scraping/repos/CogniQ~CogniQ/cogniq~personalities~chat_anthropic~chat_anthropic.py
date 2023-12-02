from __future__ import annotations
from typing import *
import logging

logger = logging.getLogger(__name__)

from haystack.nodes.prompt.invocation_layer import AnthropicClaudeInvocationLayer

from cogniq.config import ANTHROPIC_API_KEY
from cogniq.personalities import BasePersonality
from cogniq.slack import CogniqSlack


class ChatAnthropic(BasePersonality):
    @property
    def description(self) -> str:
        return "I do not modify the query. I simply ask the question to Anthropic Claude."

    @property
    def name(self) -> str:
        return "Anthropic Claude"

    async def history(self, *, event: Dict[str, str], context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Returns the history of the event.
        """
        return await self.cslack.anthropic_history.get_history(event=event, context=context)

    async def ask(
        self,
        *,
        q: str,
        message_history: List[Dict[str, str]],
        context: Dict[str, Any],
        stream_callback: Callable[..., None] | None = None,
        reply_ts: str | None = None,
        thread_ts: str | None = None,
    ) -> Dict[str, Any]:
        # disregard provided message_history and fetch from cslack
        message_history = await self.history(event=context["event"], context=context)

        stream_callback_set = stream_callback is not None
        kwargs = {
            "model": "claude-2",
            "max_tokens_to_sample": 100000,
            "temperature": 1,
            "top_p": -1,  # disabled
            "top_k": -1,
            "stop_sequences": ["\n\nHuman: "],
            "stream": stream_callback_set,
            "stream_handler": stream_callback,
        }

        api_key = ANTHROPIC_API_KEY
        layer = AnthropicClaudeInvocationLayer(api_key=api_key, **kwargs)
        newprompt = f"{message_history}\n\nHuman: {q}"
        res = layer.invoke(prompt=newprompt)

        logger.info(f"res: {res}")
        answer = "".join(res)
        return {"answer": answer, "response": res}
