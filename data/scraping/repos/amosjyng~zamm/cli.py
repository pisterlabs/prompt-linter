import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio
import glob
import os
import sys
from enum import Enum
from typing import Callable, List, Optional

import typer
import vcr_langchain as vcr
import yaml
from appdirs import AppDirs
from langchain.llms.base import BaseLLM
from langchain_contrib.utils import current_directory
from ulid import ULID
from vcr.record_mode import RecordMode

from zamm.agents.employee import ZammEmployee
from zamm.chains.ask_task import AskForTaskChain
from zamm.llms.chat import new_openai
from zamm.llms.human import ZHuman
from zamm.tasks import ADD_TYPES_AND_DOCUMENT_TEMPLATE
from zamm.utils import read_documentation

DOCUMENTATION_PATH = "documentation.zamm.md"
VISUALIZE = True
app_dirs = AppDirs(appname="zamm")
ZAMM_SESSION_PATH = app_dirs.user_data_dir + "/sessions"


app = typer.Typer(pretty_exceptions_show_locals=False)


def visualize_or_run(fn: Callable):
    # todo: change ICE to allow for running from other modules
    fn.__module__ = "__main__"

    if VISUALIZE:
        # ICE has its own defopts
        sys.argv = sys.argv[:1]
        return langchain_visualizer.visualize(fn)
    else:
        return asyncio.get_event_loop().run_until_complete(fn())


def run_chain(cassette_path: str, run: Callable):
    with vcr.use_cassette(
        path=cassette_path,
        record_mode=RecordMode.NEW_EPISODES,
    ):
        with current_directory():
            try:
                visualize_or_run(run)
            except RuntimeError:
                print(f"Exiting, session recording should be saved to {cassette_path}")


def play_interactions(
    llm: BaseLLM,
    cassette_path: str,
    tutorial_output_path: str,
):
    async def run():
        task = AskForTaskChain(llm=llm)({})
        # always trust shell commands from the person who invoked this program in the
        # first place
        employee = ZammEmployee(llm=llm, terminal_safe_mode=False)
        result = employee(task)
        if result is not None:
            with open(tutorial_output_path, "w") as f:
                try:
                    f.write(result["output"])
                    print(f"Tutorial saved to {tutorial_output_path}")
                except Exception as e:
                    print(f"Could not save tutorial to {tutorial_output_path}: {e}")

    run_chain(cassette_path, run)


def execute_llm_task(
    employee: ZammEmployee,
    task: str,
    tutorial: Optional[str],
    cassette_path: str,
):
    async def run():
        args = {"task": task}
        if tutorial is not None:
            args["documentation"] = tutorial
        result = employee(args)
        if result is not None:
            print(
                "LLM indicates it has completed the task. Session recording should be "
                f"saved to {cassette_path}"
            )

    run_chain(cassette_path, run)


SESSION_RECORD_OPTION = typer.Option(
    None,
    help="Recorded interactions from a previous unfinished session.",
)

OUTPUT_OPTION = typer.Option(
    None,
    help="File to output the tutorial to.",
)


def get_cassette_path(cassette: Optional[typer.FileText]):
    if cassette is None:
        os.makedirs(ZAMM_SESSION_PATH, exist_ok=True)
        return f"{ZAMM_SESSION_PATH}/session_{ULID()}.yaml"
    else:
        cassette.close()
        return cassette.name


def get_cassette_and_output(
    cassette: Optional[typer.FileText],
    output: Optional[typer.FileTextWrite],
):
    cassette_path = get_cassette_path(cassette)

    if output is None:
        session_prefix = os.path.splitext(cassette_path)[0]
        output_path = f"{session_prefix}.md"
    else:
        output_path = output.name
        output.close()

    return cassette_path, output_path


def get_last_session():
    # https://stackoverflow.com/a/39327156
    files = glob.glob(f"{ZAMM_SESSION_PATH}/session_*.yaml")
    if len(files) == 0:
        print("No sessions found.")
        sys.exit(1)
    return max(files, key=os.path.getctime)


@app.command()
def teach(
    session_recording: Optional[typer.FileText] = SESSION_RECORD_OPTION,
    last_session: bool = typer.Option(
        False,
        help="The last session that was in progress",
    ),
    output: Optional[typer.FileTextWrite] = OUTPUT_OPTION,
):
    """Record a tutorial interaction."""

    cassette_path, output_path = get_cassette_and_output(
        cassette=session_recording, output=output
    )

    if last_session:
        cassette_path = get_last_session()

    llm = ZHuman()
    play_interactions(
        llm=llm, cassette_path=cassette_path, tutorial_output_path=output_path
    )


@app.command()
def re_record(
    session_recording: Optional[typer.FileText] = typer.Option(
        None,
        help="Recorded interactions from a previous unfinished session.",
    ),
    last_session: bool = typer.Option(
        False,
        help="The last session that was in progress",
    ),
    output: Optional[typer.FileTextWrite] = OUTPUT_OPTION,
):
    """Re-record a tutorial interaction.

    Keep all inputs the same. Useful for when you're making cosmetic changes to the
    prompting, but wish to otherwise keep everything the same.
    """
    if not session_recording and not last_session:
        print("You must either specify session-recording or use --last-session.")
        sys.exit(1)

    cassette_path, output_path = get_cassette_and_output(
        cassette=session_recording, output=output
    )

    if last_session:
        cassette_path = get_last_session()

    try:
        with open(cassette_path) as c:
            interactions = yaml.load(c, Loader=yaml.Loader)["interactions"]
        inputs = [
            i["response"]
            for i in interactions
            if i["request"]["uri"].startswith("tool://ZHuman")
        ]
        llm = ZHuman(sequenced_responses=inputs)
        os.remove(cassette_path)
        play_interactions(llm, cassette_path, tutorial_output_path=output_path)
    except yaml.YAMLError as exc:
        print(exc)


class Safety(str, Enum):
    # todo: use Python's Flag instead
    on = "on"
    off = "off"


@app.command()
def execute(
    task: str = typer.Option(
        ...,
        help="What you'd like the LLM to do",
    ),
    documentation: Optional[str] = typer.Option(
        None,
        help=(
            "Documentation file to help the LLM accomplish the task. Prefix with "
            "@internal for internal help files."
        ),
    ),
    session_recording: Optional[typer.FileText] = SESSION_RECORD_OPTION,
    last_session: bool = typer.Option(
        False,
        help="The last session that was in progress",
    ),
    condense_memory: bool = typer.Option(
        False,
        help=(
            "Condense agent memory during in-progress execution. Saves on tokens but "
            "may result in decreased performance."
        ),
    ),
    think_before_acting: bool = typer.Option(
        True,
        help="Asks LLM to think before acting. Increases performance but is costly.",
    ),
    model: str = typer.Option(
        "text-davinci-003",
        help="What OpenAI large language model to use for execution",
    ),
    safety: Safety = typer.Option(
        Safety.on,
        help=(
            "If on, will ask user to confirm every terminal command. If off, will run "
            "LLM commands automatically, WHICH MAY EXPOSE YOU TO LLM ATTACKS."
        ),
    ),
):
    """Ask the LLM to do something."""

    cassette_path = get_cassette_path(cassette=session_recording)
    if last_session:
        cassette_path = get_last_session()

    llm = new_openai(model_name=model, temperature=0, max_tokens=-1)
    if session_recording is not None:
        session_recording.close()
    if documentation is None:
        tutorial = None
    else:
        tutorial = read_documentation(documentation)
    employee = ZammEmployee(
        llm=llm,
        condense_memory=condense_memory,
        terminal_safe_mode=safety.value == "on",
        think_before_acting=think_before_acting,
    )
    execute_llm_task(
        employee=employee, task=task, tutorial=tutorial, cassette_path=cassette_path
    )


@app.command()
def prompt(
    raw: Optional[typer.FileText] = typer.Option(
        ...,
        help="Raw prompt to be sent to the LLM verbatim",
    ),
    stop: List[str] = typer.Option(
        [],
        help="LLM stopping tokens",
    ),
    model: str = typer.Option(
        "text-davinci-003",
        help="What OpenAI large language model to use for execution",
    ),
):
    """Test a single verbatim prompt. Good for iterative prompt development."""
    llm = new_openai(model_name=model, temperature=0, max_tokens=-1)
    escaped_stops = [s.encode("utf-8").decode("unicode_escape") for s in stop]

    async def run():
        result = llm(prompt=raw.read(), stop=escaped_stops)
        print(result)

    visualize_or_run(run)


@app.command()
def deflake(
    code: Optional[typer.FileText] = typer.Option(
        ...,
        help="Code file for the LLM to type",
    ),
    model: str = typer.Option(
        "code-davinci-002",
        help="What OpenAI large language model to use for execution",
    ),
):
    """Add types and documentation to the given file."""
    llm = new_openai(model_name=model, temperature=0, max_tokens=-1)

    async def run():
        code_content = code.read()
        code.close()
        result = llm(
            prompt=ADD_TYPES_AND_DOCUMENT_TEMPLATE.format(code=code_content), stop="```"
        )
        with open(code.name, "w") as f:
            f.write(result)
        return result

    visualize_or_run(run)
