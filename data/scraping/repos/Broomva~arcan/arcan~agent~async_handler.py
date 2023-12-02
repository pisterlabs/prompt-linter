from typing import Any

from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.schema import LLMResult


class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    content: str = ""
    final_answer: bool = False

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.content += token

        # Checking for the "Final Answer" action.
        if not self.final_answer and '"action": "Final Answer"' in self.content:
            self.final_answer = True

        # If inside the "Final Answer" action, start collecting tokens for action_input.
        elif self.final_answer:
            # if token not in ['"', "}"]:
            self.queue.put_nowait(token)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # Resetting for potential subsequent uses.
        if self.final_answer:
            self.final_answer = False
            self.content = ""
            self.done.set()

    # content: str = ""
    # final_answer: bool = False

    # async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
    #     self.content += token
    #     # if we passed the final answer, we put tokens in queue
    #     if self.final_answer:
    #         if '"action_input": "' in self.content:
    #             if token not in ['"', "}"]:
    #                 self.queue.put_nowait(token)
    #     elif "Final Answer" in self.content:
    #         self.final_answer = True
    #         self.content = ""

    # async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
    #     if self.final_answer:
    #         self.content = ""
    #         self.final_answer = False
    #         self.done.set()
    #     else:
    #         self.content = ""
