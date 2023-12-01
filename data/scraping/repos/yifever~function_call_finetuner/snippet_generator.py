import openai
from typing import Any
# we can test some different templates
from snippet_builder.message_templates import (
    MESSAGE_TEMPLATE_1 as TEMPLATE,
)

def generate_snippet_from(
    function_task: str,
    model: str = "gpt-3.5-turbo-0613",
    temperature: float = 1
) -> tuple[list[dict[Any, Any] | str], dict[str, str], dict[Any, Any]]:  # (messages, completion)
    """
    Generates a function description compatible with openAI's function call API from a snippet of the function.
    Relies heavily on the comment in front of the snippet
    """
    prompt_message = TEMPLATE.format(
        FUNCTION_TASK= str(function_task),
    )
    meta = {
        "TEMPLATE": TEMPLATE,
        "FUNCTION_TASK": str(function_task),
    } 
    messages = [{"role": "user", "content": prompt_message}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return (messages, response["choices"][0], meta)
