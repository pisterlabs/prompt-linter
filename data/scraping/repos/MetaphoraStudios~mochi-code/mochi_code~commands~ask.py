"""The ask command. This command is used to ask mochi a single question."""

import argparse
import pathlib

from dotenv import dotenv_values
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from mochi_code.commands.argument_types import valid_prompt
from mochi_code.prompts.project_prompts import get_project_prompt

# Load keys for the different model backends. This needs to be setup separately.
keys = dotenv_values(".keys")


def setup_ask_arguments(parser: argparse.ArgumentParser) -> None:
    """Setup the parser with the ask command arguments.

    Args:
        parser (argparse.ArgumentParser): The parser to add the arguments to.
    """
    parser.add_argument("prompt",
                        type=valid_prompt,
                        help="Your non-empty prompt to run.")


def run_ask_command(args: argparse.Namespace) -> None:
    """Run the 'ask' command with the provided arguments."""
    # Arguments should be validated by the parser.
    ask(args.prompt)


def ask(prompt: str) -> None:
    """Run the ask command."""
    assert prompt and prompt.strip()

    llm = OpenAI(streaming=True,
                 callbacks=[StreamingStdOutCallbackHandler()],
                 temperature=0.9,
                 openai_api_key=keys["OPENAI_API_KEY"])  # type: ignore

    template = PromptTemplate(
        input_variables=["project_prompt", "user_prompt"],
        template="You are an great software engineer helping other " +
        "engineers. Whenever possible provide code examples, prioritise " +
        "copying code from the following prompt (if available). If you're " +
        "creating a function or command, please show how to call it.\nIt's " +
        "very important you keep answers related to code, if you think the " +
        "query is not related to code, please ask to clarify, to provide " +
        "more context or rephrase the query.\nKeep answers concise and if " +
        "you don't know the answer, please say so.\nALWAYS address the user " +
        "directly, as an interactive assistant, but no need to greet, go " +
        "straight to the point, politely and very light humour when " +
        "appropriate. Do not ask follow-up questions!\n{project_prompt}\n\n" +
        "User query: '{user_prompt}'",
    )
    chain = LLMChain(llm=llm, prompt=template)

    current_path = pathlib.Path.cwd()
    project_prompt = get_project_prompt(current_path)
    chain.run(user_prompt=prompt, project_prompt=project_prompt)
