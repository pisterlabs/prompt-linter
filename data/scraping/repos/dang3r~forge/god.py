from typing import Any
import pathlib
import functools
import os
import openai

from inspect import signature


def write_func(name: str, signature: str, func_folder="functions"):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a brilliant python software engineer",
            },
            {
                "role": "user",
                "content": f"Write a python function with the following signature. Add any arguments and keyword arguments you see fit Print only the function encapsualted in a codeblock. If the function is empty, fill it in! The signature is below\n\n {signature}",
            },
        ],
    )
    func = response["choices"][0]["message"]["content"]
    parts = func.split("```")
    func = "\n".join(parts[1].split("\n")[1:])
    function_file = pathlib.Path(func_folder) / (name + ".py")
    with open(str(function_file), "w") as f:
        f.write(func)


class _God:
    def __init__(self, function_folder="functions"):
        self.function_folder = pathlib.Path(function_folder)
        if not self.function_folder.exists():
            self.function_folder.mkdir()

    def __getattribute__(self, name: str) -> Any:
        if name == "wrap" or name == "function_folder":
            return object.__getattribute__(self, name)

        def inner(*args, **kwargs):
            function_file = self.function_folder / (name + ".py")
            if not function_file.exists():
                args = ",".join(f"{k}: {type(v)}" for k, v in kwargs.items())
                signature = f"def {name}({args}):"
                write_func(name, signature)
            func = getattr(getattr(__import__("functions." + name), name), name)
            return func(*args, **kwargs)

        return inner

    def wrap(self, func):
        name = func.__name__
        function_signature = (
            f'def {func.__name__}{signature(func)}:\n"""{func.__doc__}"""\n\tpass'
        )

        @functools.wraps(func)
        def foo(*args, **kwargs):
            if not "functions." + name + ".py" in os.listdir():
                write_func(name, function_signature)
            f = getattr(getattr(__import__("functions." + name), name), name)
            return f(*args, **kwargs)

        return foo


God = _God()

print(
    God.filter_for_strings_with_us_states_in_them(
        strings=["California cheeseburger", "funny man Alaska", "dog", "canada"]
    )
)
print(God.random_canadian_province())


@God.wrap
def how_many_calories_in_this_meal(meal_description: str) -> int:
    """Given an example meal description, return the number of calories in that meal.

    Example:
        - A Large Big mac and large fries -> 1130

    Args:
        meal_description (str): Description of the meal

    Returns:
        int: The number of calories
    """
    pass


@God.wrap
def is_this_a_healthy_meal(meal_description: str) -> bool:
    """Given an example meal description, return True if the meal is healthy, False otherwise.

    Example:
        - A Large Big mac and large fries -> unhealthy

    Args:
        meal_description (str): Description of the meal

    Returns:
        bool: True if the meal is healthy, False otherwise
    """
    pass


print(
    how_many_calories_in_this_meal(
        "A cheeseburger with bacon, cheese, and two angus beef patties. I also had half a palte of calamari and a large coke."
    )
)
print(
    is_this_a_healthy_meal(
        "A cheeseburger with bacon, cheese, and two angus beef patties. I also had half a palte of calamari and a large coke."
    )
)
