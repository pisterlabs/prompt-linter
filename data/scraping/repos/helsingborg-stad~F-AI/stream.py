from planning_permission.chat.prompt import ChatPrompt
from planning_permission.chat.settings import OpenAIStreamSettings, default_settings
from langstream.contrib import OpenAIChatStream, OpenAIChatMessage
from typing import Callable, Iterable, Tuple, TypeVar

T = TypeVar('T')
U = TypeVar('U')

def create_chat_prompt(prompt_args: dict) -> ChatPrompt:
    """
    Create a ChatPrompt object.

    Args:
        prompt_args (dict): Arguments for creating a ChatPrompt object.

    Returns:
        ChatPrompt: The ChatPrompt object.
    """
    return ChatPrompt(**prompt_args)

def create_chat_stream(
    name: str,
    messages_fn: Callable[[T], Iterable[OpenAIChatMessage]],
    settings: OpenAIStreamSettings = default_settings,
) -> OpenAIChatStream[T, U]:
    """
    Create a chat stream.

    Args:
        name (str): The name of the chat stream.
        settings (OpenAIStreamSettings): Settings for the chat stream.
        input_map_fn (Callable[[T], U], optional): Function that maps input of type T to output of type U. Defaults to identity function.

    Returns:
        OpenAIChatStream[T, U]: The chat stream.
    """
    return OpenAIChatStream[T, U](
        name,
        lambda delta: [*messages_fn(delta)],
        **settings
    )

def create_chat_stream_from_prompt(
    prompt_args: dict,
) -> Tuple[OpenAIChatStream[T, U], ChatPrompt]:
    """
    Create a general chat stream with a prompt.

    Args:
        prompt_args (dict): Arguments for creating a ChatPrompt object.
        settings (OpenAIStreamSettings): Settings for the chat stream.
        history (list[OpenAIChatMessage], optional): Chat history. Defaults to an empty list.
        input_map_fn (Callable[[T], U], optional): Function that maps input of type T to output of type U. Defaults to identity function.

    Returns:
        tuple[OpenAIChatStream[T, U], ChatPrompt]: A tuple containing the chat stream and prompt objects.
    """

    input_map_fn = prompt_args["input_map_fn"] if "input_map_fn" in prompt_args else lambda x: x

    prompt = create_chat_prompt({
        "name": prompt_args["name"],
        "messages": prompt_args["messages"],
        "settings": prompt_args["settings"] if "settings" in prompt_args else default_settings,
    })
    
    def messages(p: T) -> Iterable[OpenAIChatMessage]:
        prompt.format_prompt(input_map_fn(p))
        return prompt.to_messages()

    chat_stream = create_chat_stream(
        prompt.name, 
        messages,
        prompt.settings
    )
    return chat_stream, prompt