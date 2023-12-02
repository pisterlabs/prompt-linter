"""
A toy numerical example showing how two agents can collaborate on a task.

The Student Agent is tasked with calculating the sum of a list of numbers,
and is told that it knows nothing about addition, and can ask for help
from an Adder Agent who can add pairs of numbers.

Run as follows:

python3 examples/quick-start/two-agent-chat-num.py

For more explanation see the
[Getting Started guide](https://langroid.github.io/langroid/quick-start/two-agent-chat-num/)
"""

import typer

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging


app = typer.Typer()

setup_colored_logging()


def chat() -> None:
    config = ChatAgentConfig(
        llm = OpenAIGPTConfig(
            chat_model=OpenAIChatModel.GPT4,
        ),
        vecdb = None,
    )
    student_agent = ChatAgent(config)
    student_task = Task(
        student_agent,
        name = "Student",
        system_message="""
        You will receive a list of numbers from me (the User),
        and your goal is to calculate their sum.
        However you do not know how to add numbers.
        I can help you add numbers, two at a time, since
        I only know how to add pairs of numbers.
        Send me a pair of numbers to add, one at a time, 
        and I will tell you their sum.
        For each question, simply ask me the sum in math notation, 
        e.g., simply say "1 + 2", etc, and say nothing else.
        Once you have added all the numbers in the list, 
        say DONE and give me the final sum. 
        Start by asking me for the list of numbers.
        """,
        llm_delegate=True,
        single_round=False,
    )
    adder_agent = ChatAgent(config)
    adder_task = Task(
        adder_agent,
        name = "Adder",
        system_message="""
        You are an expert on addition of numbers. 
        When given numbers to add, simply return their sum, say nothing else
        """,
        single_round=True,  # task done after 1 step() with valid response
    )
    student_task.add_sub_task(adder_task)
    student_task.run()


@app.command()
def main(
        debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
        no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
        nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
        )
    )
    chat()


if __name__ == "__main__":
    app()