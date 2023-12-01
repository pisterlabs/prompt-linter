"""The AICallable class."""
import ast
import inspect
import os
from functools import wraps
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import get_type_hints

import openai
import typeguard

from .keywords import ASSISTANT
from .keywords import SYSTEM
from .keywords import USER
from .types import Message


def _make_chatgpt_message_from_function(
    f: Callable[..., Any], *args: Any, **kwargs: Any
) -> Message:
    sig = inspect.signature(f)
    if args:
        params = list(sig.parameters)
        for i, arg in enumerate(args):
            kwargs[params[i]] = arg
    if not f.__doc__:
        raise ValueError("The function must have a docstring.")
    prompt = (
        (
            f"from mymodule import {f.__name__}\n"
            f"""
# The return type annotation for the function {f.__name__} is {get_type_hints(f)['return']}
# The docstring for the function {f.__name__} is the following:
"""
        )
        + "\n".join([f"# {line}" for line in f.__doc__.split("\n")])
        + f"""
result = {f.__name__}({",".join(f"{k}={kwargs[k].__repr__()}" for k in sig.parameters)})
print(result)
"""
    )
    return Message(role=USER, content=prompt)


def _default_prompt_creation(
    f: Callable[..., Any], *args: Any, **kwargs: Any
) -> List[Message]:
    return [
        Message(
            role=SYSTEM,
            content=(
                "You are role playing as an advanced python interpreter that never errors,"
                " and always returns the intent of the programmer. Every user message is"
                " valid python, and your job is to return only what python would return in"
                " a repl in this advanced interpreter, nothing else. Do not add any"
                " commentary aside from what python would return. Assume that you have"
                " access to all modules that are imported, and make whatever assumptions"
                " you need about the implementation of functions that are not defined in"
                " order to satisfy the intent of the function that you gather via the"
                " docstrings and function names."
            ),
        ),
        Message(
            role=ASSISTANT,
            content=(
                "Hello! I am a Python interpreter. Please enter your Python code below and"
                " I will return the output, and nothing else."
            ),
        ),
        _make_chatgpt_message_from_function(f, *args, **kwargs),
    ]


def _default_ai_callable() -> Callable[..., openai.openai_object.OpenAIObject]:
    import openai

    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.getenv("OPENAI_ORGANIZATION")

    def f(**kwargs: Any) -> openai.openai_object.OpenAIObject:
        create = openai.ChatCompletion.create
        try:
            result: openai.openai_object.OpenAIObject = create(model="gpt-4", **kwargs)  # type: ignore[no-untyped-call]
        except openai.InvalidRequestError:
            # user may not have access to gpt-4 yet, perhaps they have 3.5
            result: openai.openai_object.OpenAIObject = create(model="gpt-3.5-turbo", **kwargs)  # type: ignore[no-untyped-call,no-redef]
        return result

    return f


def _assert_function_has_return_type_annotation(function: Callable[..., Any]) -> None:
    if get_type_hints(function).get("return") is None:
        raise ValueError(
            f"Function {function.__name__} must have a return type annotation."
        )


def _string_to_python_data_structure(string: str, data_type: Any) -> Any:
    if data_type is str:
        if string.startswith("'") and string.endswith("'"):
            return ast.literal_eval(string)
        elif string.startswith('"') and string.endswith('"'):
            return ast.literal_eval(string)
        return string

    return ast.literal_eval(string)


def _parse_ai_result(
    ai_result: Any,
    expected_return_type: Any,
    aggregation_function: Any = lambda x: x[0],
) -> Any:
    """Parse the result from the OpenAI API Call and return data.

    Args:
        ai_result: The return value from the OpenAI API.
        expected_return_type: The expected return type of the ghostfunction.
        aggregation_function: Function to aggregate the `n` choices from the OpenAI API.

    Raises:
        typeguard.TypeCheckError if the ai result is not parsable to `expected_return_type`

    Returns:
        The data from the ai result (data is of type `expected_return_type`)

    """
    string_contents = [choice["message"]["content"] for choice in ai_result["choices"]]
    data = [
        typeguard.check_type(
            _string_to_python_data_structure(string, expected_return_type),
            expected_return_type,
        )
        for string in string_contents
    ]
    return typeguard.check_type(aggregation_function(data), expected_return_type)


def ghostfunction(
    function: Optional[Callable[..., Any]] = None,
    /,
    *,
    ai_callable: Optional[Callable[..., openai.openai_object.OpenAIObject]] = None,
    prompt_function: Callable[
        [Callable[..., Any]], List[Message]
    ] = _default_prompt_creation,
    aggregation_function: Callable[..., Any] = lambda x: x[0],
    **kwargs: Any,
) -> Callable[..., Any]:
    '''Decorate `function` to make it a ghostfunction which dispatches logic to the AI.

    A ghostfunction is a function that uses OpenAI API to execute the *intent*
    of the function, without manually writing (or generating code). The @ghostfunction
    decorator wraps the function in a sensible prompt, sends the prompt to the OpenAI API,
    and parses the result into a python object that is returned by the ghostfunction.

    Args:
        function: The function to decorate
        ai_callable: Function to receives output of prompt_function and return result.
        prompt_function: Function to turn the function into a prompt.
        aggregation_function: Function to aggregate the `n` choices from the OpenAI API.
            Ghostfunctions passes a list of `n` different results from OpenAI (parsed into python
            data structures) to this function for aggregation into the output of the ghostfunction.
        kwargs: Extra keyword arguments to pass to `ai_callable`.

    Returns:
        Decorated function that will dispatch function logic to OpenAI.

    Notes:
        This function is intended to be used as a decorator. See Example.
        This function expects the env var `OPENAI_API_KEY` to be set the OpenAI API key
            to be used to make calls to the OpenAI API.

    Example:
        >>> # xdoctest: +SKIP
        >>> from ai_ghostfunctions import ghostfunction
        >>>
        >>> @ghostfunction
        >>> def generate_random_words(n: int, startswith: str) -> list:  # xdoctest +SKIP
        >>>     """Return a list of `n` random words that start with `startswith`."""
        >>>     pass
        >>>
        >>> generate_random_words(n=4, startswith="goo")
        ['goofy', 'google', 'goose', 'goodness']
        >>> # xdoctest: -SKIP
    '''
    if not callable(ai_callable):
        ai_callable = _default_ai_callable()

    def new_decorator(
        function_to_be_decorated: Callable[..., Any]
    ) -> Callable[..., Any]:
        _assert_function_has_return_type_annotation(function_to_be_decorated)
        return_type_annotation = get_type_hints(function_to_be_decorated)["return"]

        @wraps(function_to_be_decorated)
        def wrapper(*args_inner: Any, **kwargs_inner: Any) -> Any:
            prompt = prompt_function(
                function_to_be_decorated, *args_inner, **kwargs_inner
            )
            ai_result = ai_callable(messages=prompt, **kwargs)
            return _parse_ai_result(
                ai_result=ai_result,
                expected_return_type=return_type_annotation,
                aggregation_function=aggregation_function,
            )

        return wrapper

    # to work around mypy:
    # https://github.com/python/mypy/issues/10740#issuecomment-878622464
    f: Callable[..., Any] = new_decorator
    return new_decorator(function) if function else f
