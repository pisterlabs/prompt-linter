from aitemplates.oai.utils.wrappers import retry_openai_api
import os
from dotenv import load_dotenv

from typing import Any, Optional, List, Union

import aiolimiter
import openai
import openai.error
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio

from aitemplates.oai.types.base import ResponseDict
from aitemplates.oai.types.chat import ChatConversation, ChatPair
from aitemplates.oai.ApiManager import SingleApiManager

dotenv_path = os.path.join(os.getcwd(), '.env')  # get the path to .env file in current working directory
load_dotenv(dotenv_path)  # load environment variables from the .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = os.getenv("model")

if OPENAI_API_KEY is None:
    raise Exception("API key not found in environment variables")

openai.api_key = OPENAI_API_KEY


@retry_openai_api()
async def _throttled_acreate_chat_completion(
    chat_pair: ChatPair,
    limiter: aiolimiter.AsyncLimiter,
    **kwargs
) -> Any:
    """Create a throttled chat completion using the OpenAI API.

    This function ensures the number of requests made to the OpenAI API does not exceed a specified limit.

    Args:
        messages (list[MessageDict]): The messages to send to the chat completion.
        limiter (aiolimiter.AsyncLimiter): The limiter that manages throttling of requests.
        model (str, optional): The model to use. Defaults to "gpt-3.5-turbo".
        temperature (float, optional): The temperature to use. Defaults to 0.
        max_tokens (int, optional): The maximum tokens to use. Defaults to None.
        top_p (float, optional): The nucleus sampling probability. Defaults to 1.
        n (int, optional): The number of messages to generate. Defaults to 1.
        stop (str, optional): The sequence at which the generation will stop. Defaults to None.
        presence_penalty (float, optional): The presence penalty to use. Defaults to 0.
        frequency_penalty (float, optional): The frequency penalty to use. Defaults to 0.

    Returns:
        Any: The response from the chat completion.
    """
    async with limiter:
        response = await openai.ChatCompletion.acreate(
            messages=chat_pair.prompt_raw,
            **kwargs
        )
        return chat_pair, response


async def async_create_chat_completion(
    messages: ChatConversation,
    model: str = model or "gpt-3.5-turbo",
    temperature: float = 0,
    max_tokens: Union[int, None] = None,
    print_every: int = False,
    keep_order: bool = False,
    requests_per_minute: int = 300,
    top_p: Optional[float] = 1,
    n: Optional[int] = 1,
    stop: Optional[str] = None,
    presence_penalty: Optional[float] = 0,
    frequency_penalty: Optional[float] = 0,
) -> ChatConversation:
    """Generate from OpenAI Chat Completion API asynchronously.

    Args:
        messages (ChatConversation): List of full prompts to generate from.
        model (str, optional): Model configuration. Defaults to "gpt-3.5-turbo".
        temperature (float, optional): Temperature to use. Defaults to 0.
        max_tokens (Union[int, None], optional): Maximum number of tokens to generate. Defaults to None.
        print_every (int, optional): Print the response after every this many calls. It does nothing if keep_order is True. Defaults to False.
        keep_order (bool, optional): If True, keeps the order of responses same as the input prompts. Defaults to False.
        requests_per_minute (int, optional): Number of requests per minute to allow. Defaults to 300.
        response_list (List[str], optional): If provided, responses are added to this list.
        top_p (float, optional): The nucleus sampling probability. Defaults to 1.
        n (int, optional): The number of messages to generate. Defaults to 1.
        stop (str, optional): The sequence at which the generation will stop. Defaults to None.
        presence_penalty (float, optional): The presence penalty to use. Defaults to 0.
        frequency_penalty (float, optional): The frequency penalty to use. Defaults to 0.

    Returns:
        ChatConversation: List of generated responses and previous responses.
    """
    if keep_order and print_every:
        print("print_every will do nothing since keep_order is True")
    api_manager = SingleApiManager()
    openai.aiosession.set(ClientSession())
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    
    kwargs = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "n": n,
        "stop": stop,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
    }
    
    async_responses = [
        _throttled_acreate_chat_completion(
            **kwargs,
            limiter=limiter,
            chat_pair=chat_pair,
        )
        for chat_pair in messages.conversation_history
    ]

    responses = []
    prompt_tokens = 0
    completion_tokens = 0

    if keep_order:
        # Keep the order of responses same as input prompts
        collect_responses = await tqdm_asyncio.gather(*async_responses)

        # Update the cost associated with this response in the API manager
        for i, response in enumerate(collect_responses):
            chat_pair, openai_response = response
            responses.append(chat_pair.update_response(ResponseDict(openai_response)))
            prompt_tokens += openai_response.usage.prompt_tokens
            completion_tokens += openai_response.usage.completion_tokens

            if i == len(collect_responses) - 1:
                api_manager.update_cost(
                    prompt_tokens,
                    completion_tokens,
                    openai_response.model,
                )
    else:
        # Without order
        for future in tqdm_asyncio.as_completed(async_responses):
            collect_response = await future
            
            chat_pair, openai_response = collect_response

            # Update the cost associated with this response in the API manager
            api_manager.update_cost(
                openai_response.usage.prompt_tokens,
                openai_response.usage.completion_tokens,
                openai_response.model,
            )

            response_text = openai_response.choices[0].message.content
            responses.append(chat_pair.update_response(ResponseDict(openai_response)))
            if print_every:
                print(response_text + "\n")

    # Close the session
    await openai.aiosession.get().close()  # type: ignore
    return messages.fill_conversation(responses)
