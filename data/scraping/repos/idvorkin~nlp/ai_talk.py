#!python3

import os
import json
from icecream import ic
import typer
from rich.console import Console
from rich import print
from typing import List
from pydantic import BaseModel
from loguru import logger
import pudb
from typing_extensions import Annotated

console = Console()
app = typer.Typer()
from langchain.llms import GPT4All
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import Any, Optional
from langchain.output_parsers.openai_functions import OutputFunctionsParser
from langchain.schema import FunctionMessage


from langchain.schema import (
    Generation,
    OutputParserException,
)


def model_to_function(cls):
    return {"name": cls.__name__, "parameters": cls.model_json_schema()}


class JsonOutputFunctionsParser2(OutputFunctionsParser):
    """Parse an output as the Json object."""

    def parse_result(self, result: List[Generation]) -> Any:
        function_call_info = super().parse_result(result)
        if self.args_only:
            try:
                # Waiting for this to merge upstream
                return json.loads(function_call_info, strict=False)
            except (json.JSONDecodeError, TypeError) as exc:
                raise OutputParserException(
                    f"Could not parse function call data: {exc}"
                )
        function_call_info["arguments"] = json.loads(function_call_info["arguments"])
        return function_call_info


# Todo consider converting to a class
class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Shared command line arguments
# https://jacobian.org/til/common-arguments-with-typer/
@app.callback()
def load_options(
    ctx: typer.Context,
    attach: bool = Annotated[bool, typer.Option(prompt="Attach to existing process")],
):
    ctx.obj = SimpleNamespace(attach=attach)


# Our task is the docstring of the called function
# We're called from a helper function this task so we're 2 down in the stack
def tell_our_task():
    import inspect

    for frame_info in inspect.stack()[2:3]:
        function_name = frame_info.function
        function_object = frame_info.frame.f_globals.get(function_name)
        if function_object:
            docstring = inspect.getdoc(function_object)
            print(f"Our task is: {docstring}")
            input()


def process_shared_app_options(ctx: typer.Context):
    if ctx.obj.attach:
        pudb.set_trace()
    tell_our_task()


@logger.catch()
def app_wrap_loguru():
    app()


def tell_model_ready():
    print("ready to show output")
    input()
    print("--")


@app.command()
def talk_1(ctx: typer.Context, topic: str = "software engineers", count: int = 2):
    """Input to output: Get  a joke"""
    process_shared_app_options(ctx)
    tell_our_task()

    model = ChatOpenAI()
    prompt = ChatPromptTemplate.from_template("tell me {count} jokes about {topic}")
    print(prompt.messages)
    chain = prompt | model
    response = chain.invoke({"topic": topic, "count": count})
    tell_model_ready()
    print(response.content)


@app.command()
def talk_2(ctx: typer.Context, topic: str = "software engineers", count: int = 2):
    """Chain: Get  a joke, and why its funny"""

    process_shared_app_options(ctx)

    class Joke(BaseModel):
        setup: str
        punch_line: str
        reason_joke_is_funny: str

    class GetJokes(BaseModel):
        count: int
        jokes: List[Joke]

    model = ChatOpenAI()
    prompt = ChatPromptTemplate.from_template("tell me {count} jokes about {topic}")
    chain = (
        prompt
        | model.bind(functions=[model_to_function(GetJokes)])
        | JsonOutputFunctionsParser2()
    )
    print(prompt.messages)
    response = chain.invoke({"topic": topic, "count": count})
    tell_model_ready()
    print(response)


@app.command()
def talk_3(ctx: typer.Context, n: int = 20234, count: int = 4):
    """Get the n-th prime"""
    process_shared_app_options(ctx)

    print("FYI: Like humans, models hallucinate ")
    prompt = ChatPromptTemplate.from_template(f"What is the {n}th prime")
    model = ChatOpenAI()
    chain = prompt | model

    for _ in range(count):
        response = chain.invoke({})
        ic(response)


@app.command()
def talk_4(ctx: typer.Context, n: int = 20234):
    """Get the nth prime, but use tools"""
    process_shared_app_options(ctx)

    class ExecutePythonCode(BaseModel):
        valid_python: str
        code_explanation: str

    model = ChatOpenAI(model="gpt-4-0613").bind(
        function_call={"name": "ExecutePythonCode"},  # tell gpt to use this model
        functions=[model_to_function(ExecutePythonCode)],
    )

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "Write code to solve the users problem. the last line of the python  program should print the answer. Do not use sympy"
            ),
            HumanMessagePromptTemplate.from_template("What is the {n}th prime"),
        ]
    )
    print(prompt.messages)

    chain = prompt | model | JsonOutputFunctionsParser2()
    response = chain.invoke({"n": n})
    input("Ready")
    print("----")

    valid_python = response["valid_python"]
    print(valid_python)
    print("----")
    print(response["code_explanation"])
    print("----")
    input("Are you sure you want to run this code??")
    exec(valid_python)


@app.command()
def talk_5(
    ctx: typer.Context,
    topic: str = "software engineers",
    count: int = 2,
    season: str = "winter",
):
    """Tell me a joke, according to the season"""
    process_shared_app_options(ctx)

    class Joke(BaseModel):
        setup: str
        punch_line: str
        reasoning_for_joke: str

    class Jokes(BaseModel):
        count: int
        jokes: List[Joke]

    class GetCurrentSeason(BaseModel):
        pass

    model = ChatOpenAI()
    # model = ChatOpenAI(model="gpt-4")
    model = model.bind(
        functions=[model_to_function(Jokes), model_to_function(GetCurrentSeason)]
    )

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a great comedian. You know it's critical to tell joke related to the season "
            ),
            HumanMessagePromptTemplate.from_template(
                "tell me {count} jokes about {topic} take into consideration the current season"
            ),
        ]
    )

    for i in range(4):  # Keep it limited to 1000 to avoid a run away loop
        chain = prompt | model
        print(prompt.messages)
        input("Show Model Output")
        response = chain.invoke({"topic": topic, "count": count})
        called_function = response.additional_kwargs["function_call"]["name"]
        arguments = response.additional_kwargs["function_call"]["arguments"]
        match called_function:
            case "GetCurrentSeason":
                ic(called_function)
                print(f"'Calling' GetCurrentSeason returning {season}")
                # 'simulate calling the function, include it in state'
                prompt.append(FunctionMessage(name=called_function, content=season))
            case "Jokes":
                ic(called_function)
                print(arguments)
                break
            case _:
                # if it's another function process that
                ic("Sorry, I don't support {function} yet")
                break

    # JsonKeyOutputFunctionsParser(key_name="jokes")


@app.command()
def moderation(ctx: typer.Context, user_input: str = "You are stupid"):
    """Moderation"""
    process_shared_app_options(ctx)

    from langchain.chains import OpenAIModerationChain

    model = (
        OpenAI()
    )  # Sheesh, Why can't I use the chat model - so much incompatibility yet.
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "Repeat what the user says back to them"
            ),
            HumanMessagePromptTemplate.from_template(user_input),
        ]
    )
    raw_chain = prompt | model
    response = raw_chain.invoke({"user_input": user_input})
    print("Raw output")
    print(response)

    moderation = OpenAIModerationChain()

    print("Output with moderation")
    moderated_chain = raw_chain | moderation
    response = moderated_chain.invoke({"user_input": user_input})
    print(response)


@app.command()
def talk_91(ctx: typer.Context, topic: str = "software engineers", count: int = 2):
    """Input to output: Get  a joke, on device!"""
    process_shared_app_options(ctx)
    tell_our_task()

    model = GPT4All(model="./falcon.bin")
    prompt = ChatPromptTemplate.from_template("tell me {count} jokes about {topic}")
    print(prompt.messages)
    chain = prompt | model
    response = chain.invoke({"topic": topic, "count": count})
    tell_model_ready()
    print(response)


@app.command()
def docs():
    from langchain.document_loaders import DirectoryLoader

    loader = DirectoryLoader(os.path.expanduser("~/blog/_d"), glob="**/*.md")
    # docs = loader.load()
    from langchain.indexes import VectorstoreIndexCreator

    index = VectorstoreIndexCreator().from_loaders([loader])
    answer = index.query("What should a manager do")
    ic(answer)


import ast
import sys


def get_func(filename, funcname):
    with open(filename, "r") as source:
        tree = ast.parse(source.read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == funcname:
            return ast.unparse(node)
    return None


@app.command()
def dump(funcname1):
    filename = sys.argv[0]
    funcname1 = funcname1.replace("-", "_")
    func1 = get_func(filename, funcname1)

    if not func1:
        print(f"Function {funcname1} not found in {filename}")
        return
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f1:
        f1.write(func1)
        f1.write("\n")
        f1.flush()
        import subprocess

        subprocess.run(["rich", "-n", f"{f1.name}"], capture_output=False)


@app.command()
def diff(funcname1, funcname2):
    filename = sys.argv[0]
    ic(filename)
    funcname1 = funcname1.replace("-", "_")
    funcname2 = funcname2.replace("-", "_")
    ic(funcname1, funcname2)
    func1 = get_func(filename, funcname1)
    func2 = get_func(filename, funcname2)

    if not func1:
        print(f"Function {funcname1} not found in {filename}")
        return
    if not func2:
        print(f"Function {funcname2} not found in {filename}")
        return

    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False
    ) as f1, tempfile.NamedTemporaryFile(mode="w", delete=False) as f2:

        f1.write(func1)
        f1.write("\n")
        f1_name = f1.name
        f2.write(func2)
        f2_name = f2.name
        f2.write("\n")

    import subprocess

    subprocess.run(["delta", "--line-numbers", f1_name, f2_name])


if __name__ == "__main__":
    app_wrap_loguru()
