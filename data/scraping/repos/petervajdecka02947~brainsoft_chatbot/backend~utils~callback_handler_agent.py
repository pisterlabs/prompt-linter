import asyncio
from typing import Any
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.schema import LLMResult


class AsyncCallbackHandler(AsyncIteratorCallbackHandler):
    """
    A custom callback handler for asynchronous streaming of tokens from the language model.

    Attributes:
        content (str): Accumulates the tokens received from the language model.
        final_answer (bool): Flag indicating if the final answer has been reached.

    Args:
        delay (float): Delay in seconds before processing each new token. Defaults to 1.0.

    This class extends AsyncIteratorCallbackHandler and implements custom logic for handling new tokens and the end of a language model's response.
    """

    content: str = ""
    final_answer: bool = False

    def __init__(self, delay: float = 1.0) -> None:
        super().__init__()
        self.delay = delay

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await asyncio.sleep(self.delay)
        self.content += token
        # if we passed the final answer, we put tokens in queue
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ['"', "}"]:
                    self.queue.put_nowait(token)
        elif "Final Answer" in self.content:
            self.final_answer = True
            self.content = ""

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.final_answer:
            self.content = ""
            self.final_answer = False
            self.done.set()
        else:
            self.content = ""


async def run_call_no_stream(agent: object, query: str):
    """
    Executes a non-streaming call to the language model.

    Args:
        agent (object): The conversational agent object.
        query (str): The input query to be processed by the agent.

    Returns:
        The result from processing the input query by the agent.

    This function makes an asynchronous call to the agent with the given query and an empty chat history.
    """
    return await agent.acall(inputs={"input": query, "chat_history": []})


async def run_acall(agent: object, query: str, stream_it: AsyncCallbackHandler):
    """
    Runs an asynchronous call with streaming enabled using a custom callback handler.

    Args:
        agent (object): The conversational agent object.
        query (str): The input query to be processed by the agent.
        stream_it (AsyncCallbackHandler): The callback handler for streaming the response.

    This function sets the callback handler for the agent and initiates an asynchronous call with the given query.
    """
    agent.agent.llm_chain.llm.callbacks = [stream_it]
    # now query
    await agent.acall(inputs={"input": query, "chat_history": []})


async def create_gen(agent: object, query: str, stream_it: AsyncCallbackHandler):
    """
    Creates an asynchronous generator for streaming tokens from the language model.

    Args:
        agent (object): The conversational agent object.
        query (str): The input query to be processed by the agent.
        stream_it (AsyncCallbackHandler): The callback handler for streaming the response.

    Returns:
        An asynchronous generator yielding tokens from the language model.

    This function initiates an asynchronous call with streaming and yields tokens as they are received.
    """
    task = asyncio.create_task(run_acall(agent, query, stream_it))

    async for token in stream_it.aiter():
        yield token
    await task
