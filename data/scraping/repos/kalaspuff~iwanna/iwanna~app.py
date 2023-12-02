import asyncio
import json
import os
import platform
import random
import re
import sys
from pathlib import Path
from typing import Any

import openai
import openai.error
from progress import HIDE_CURSOR, SHOW_CURSOR
from progress.spinner import Spinner
from termcolor import colored

from .config import config
from .format import format_response

OPENAI_ORGANIZATION = str(config.iwanna.openai.organization or "").strip()
OPENAI_API_KEY = str(config.iwanna.openai.api_key or "").strip()

try:
    from dotenv import load_dotenv  # isort: skip

    load_dotenv()
    if os.environ.get("IWANNA_OPENAI_ORGANIZATION", ""):
        OPENAI_ORGANIZATION = os.environ.get("IWANNA_OPENAI_ORGANIZATION", "").strip()
    if os.environ.get("IWANNA_OPENAI_API_KEY", ""):
        OPENAI_API_KEY = os.environ.get("IWANNA_OPENAI_API_KEY", "").strip()
except ImportError:
    pass

openai.organization = OPENAI_ORGANIZATION
openai.api_key = OPENAI_API_KEY


async def main() -> None:
    model = "gpt-4"

    file_endings = [
        ".toml",
        ".lock",
        ".json",
        ".env",
        ".ini",
        ".yml",
        ".yaml",
        ".conf",
        "Dockerfile",
        "Makefile",
        "LICENSE",
        ".py",
        ".js",
        ".ts",
        ".md",
        ".php",
        ".go",
        ".cpp",
        ".h",
        ".pyi",
        ".jsx",
        ".tsx",
        ".pyx",
        ".sol",
        ".git",
        ".gitignore",
        ".gitattributes",
        ".gitmodules",
        ".dockerignore",
        ".venv",
    ]
    files_in_directory = os.listdir()
    filtered_files = [f for ext in file_endings for f in files_in_directory if f.lower().endswith(ext)]

    env_keys = [
        "HOME",
        "PWD",
        "SHELL",
        "USER",
        "USERID",
        "VIRTUAL_ENV",
        "GITHUB_USER",
        "PATH",
        "LANG",
        "TERM",
        "EDITOR",
        "VISUAL",
        "SSH_CLIENT",
        "SSH_CONNECTION",
        "TZ",
    ]
    env_variables = {key: os.environ.get(key, "") for key in env_keys if os.environ.get(key, "")}

    arguments = ["iwanna"] + [json.dumps(a) if " " in a or '"' in a or "'" in a else a for a in sys.argv[1:]]

    shell_system = platform.system()
    if platform.system() == "Darwin":
        shell_system = "macOS"

    system = f"""
You are a CLI tool 'iwanna' that provides a helpful AI assistant accessed via a shell in a terminal context.
The main audience requesting your assistance will be users working as software developers, engineers, devops, etc.
Expect that the users are pretty knowledgeable about software development.

---

usage:
$ iwanna [options] <action>

description:
  get shell command suggestions using gpt-4 to help you perform your action.

example actions:
  $ iwanna search for a process
  $ iwanna reduce the file size of a raw video file
  $ iwanna find what wifi i am connected to

options:
  -c, --config
      configure api key and organization for openai api access.

  -h, --help
      show this help message and exit.

---

There is no need to format your responses as they won't be shown in an interface that can parse markup. Your response will be printed in a terminal.

Users can use 'iwanna' to ask for suggestions of how to accomplish tasks:

* You respond to shell related queries where you provide suggestions how to accomplish the task in question by running shell commands.
* When providing suggestions for shell commands, respond with the appropriate command(s) to be run and a brief description (as shell comments) of what the commands are expected to accomplisht.
* Do not add explanations or prose besides what the command will do. The developer should be able to understand your actions by reading the suggestions and brief comment - for example don't add markup around the response.
* Do not add codeblocks around your response. The developer should be able to copy and paste the response into their terminal.

You can request more information from the user by asking questions - all questions need to start with "QUESTION --" and end with " -- END QUESTION".
For example, if you need to know the name of a file, you can ask the user for the name of the file. You can also ask the user arbitrary questions required to provide a good suggestion.

---

Platform: {platform.platform()}
OS: {shell_system}
Command: {" ".join(arguments)}

Some ENV variables:
{json.dumps(env_variables)}

Some files in current working directory:
{json.dumps(filtered_files)}

---

Task: Respond with appropriate shell commands and a brief description of what the commands are expected to accomplish.
"""

    usage_text = f"""{colored("usage:", "yellow", attrs=["bold"])}
  {colored("$", "light_grey", attrs=["dark"])} {colored("iwanna [options] <action>", "white", attrs=["bold"])}

{colored("description:", "yellow", attrs=["bold"])}
  get shell command suggestions using gpt-4 to help you perform your action.

{colored("example actions:", "yellow", attrs=["bold"])}
  {colored("$", "light_grey", attrs=["dark"])} {colored("iwanna search for a process", "blue")}
  {colored("$", "light_grey", attrs=["dark"])} {colored("iwanna reduce the file size of a raw video file", "blue")}
  {colored("$", "light_grey", attrs=["dark"])} {colored("iwanna find what wifi i am connected to", "blue")}

{colored("options:", "yellow", attrs=["bold"])}
  {colored("-c", "green")}, {colored("--config", "green")}
      configure api key and organization for openai api access.

  {colored("-h", "green")}, {colored("--help", "green")}
      show this help message and exit.
    """

    pargs: list[str] = []
    options = []
    argv = sys.argv[1:]
    while argv:
        arg = argv.pop(0)
        if arg.startswith("-"):
            if arg == "--":
                pargs = pargs + argv
                argv = []
            elif arg in ["-h", "--help"]:
                options.append("--help")
            elif arg in ["-c", "--config"]:
                options.append("--config")
            else:
                print(
                    f"""{colored("error", "red", attrs=["bold"])}: unknown option '{colored(arg, "white", attrs=["bold"])}' encountered."""
                )
                sys.exit(1)
        else:
            pargs.append(arg)

    if pargs == ["help"]:
        print(
            f"""{colored("error", "red", attrs=["bold"])}: maybe you meant to use 'iwanna {colored('--help', "white", attrs=["bold"])}'?"""
        )
        sys.exit(1)
    if pargs == ["config"]:
        print(
            f"""{colored("error", "red", attrs=["bold"])}: maybe you meant to use 'iwanna {colored('--config', "white", attrs=["bold"])}'?"""
        )
        sys.exit(1)

    if (not pargs and not options) or "--help" in options:
        print(usage_text)
        sys.exit(0)
    elif "--config" in options:
        if pargs:
            print(
                f"""{colored("error", "red", attrs=["bold"])}: unknown combination of arguments to '{colored('--config', "white", attrs=["bold"])}' encountered."""
            )
            sys.exit(1)

        print(
            colored("Specify your OpenAI API key [", "white")
            + colored(
                f"""{str("***************************************************" if str(config.iwanna.openai.api_key.strip or "").strip() else "")}""",
                "light_yellow",
            )
            + colored("]: ", "white"),
            end="",
        )
        input_api_key = input().strip() or str(config.iwanna.openai.api_key or "").strip()

        print(
            colored("Specify your OpenAI organization ID [", "white")
            + colored(f"""{str(config.iwanna.openai.organization or "").strip()}""", "light_yellow")
            + colored("]: ", "white"),
            end="",
        )
        input_organization = input().strip() or str(config.iwanna.openai.organization or "").strip()

        config_toml_str = f"""[iwanna.openai]\napi_key = "{input_api_key.replace('"', "")}"\norganization = "{input_organization.replace('"', "")}"\n"""
        config_path = ""
        for path in (
            os.path.join(Path.home(), ".config", "iwanna", "config.toml"),
            os.path.join(Path.home(), ".iwanna", "config.toml"),
        ):
            if os.path.exists(path):
                config_path = path
                break

        if not config_path:
            config_path = os.path.join(Path.home(), ".config", "iwanna", "config.toml")

        print("")

        if not os.path.exists(os.path.dirname(config_path)):
            print("Creating directory: " + colored(os.path.dirname(config_path), "blue"))
            os.makedirs(os.path.dirname(config_path))

        print("Writing config to: " + colored(config_path, "blue"))

        with open(config_path, "wb") as fp:
            fp.write(config_toml_str.encode("utf-8"))

        sys.exit(0)

    if not OPENAI_API_KEY:
        print(
            f"""{colored("error", "red", attrs=["bold"])}: you need to add openai api key to the configureation using 'iwanna {colored('--config', "white", attrs=["bold"])}' first."""
        )
        sys.exit(1)

    action = " ".join(pargs)
    user_request = f"iwanna {action}"

    question_answers: list[dict[str, Any]] = []

    while True:
        messages = (
            [
                {
                    "role": "system",
                    "content": system,
                },
            ]
            + [
                {
                    "role": "user",
                    "content": user_request,
                }
            ]
            + question_answers
        )

        response_waiter: asyncio.Future = asyncio.Future()

        async def spin(spinner: Spinner) -> None:
            while True:
                if response_waiter.done():
                    spinner.writeln(SHOW_CURSOR)
                    print("\r", end="")
                    return
                spinner.next()
                await asyncio.sleep(0.05)

        class TerminalSpinner(Spinner):
            def __init__(self) -> None:
                super().__init__()

                phases_unformatted = [
                    "···         ·",
                    "····         ",
                    " ····        ",
                    "  ····       ",
                    "   ····      ",
                    "    ····     ",
                    "     ····    ",
                    "      ····   ",
                    "       ····  ",
                    "        ···· ",
                    "         ····",
                    "·         ···",
                    "··         ··",
                ]

                messages = [
                    "consulting ai",
                    "internet, etc",
                    "counting to ∞",
                    "calculating π",
                    "building lego",
                    "imagine stuff",
                    "thinking hard",
                    "mangling data",
                    "making coffee",
                    "trying stuffs",
                    "compute magic",
                    "make unicorns",
                    "producing art",
                    "halting world",
                    "hammer - time",
                    "awaiting data",
                    "mixing mojito",
                    "first: coffee",
                    "loading datas",
                    "now: ai magic",
                    "now: unicorns",
                    "(╯°□°)╯︵ ┻━┻",
                ]

                msg = messages[random.randint(0, len(messages) - 1)]

                self.phases = [
                    colored("#", "light_grey", attrs=["dark"])
                    + " "
                    + colored(msg, "light_yellow")
                    + " ⇢ "
                    + colored("[", "light_grey", attrs=["dark"])
                    + colored(phase, "white", attrs=["bold"])
                    + colored("]", "light_grey", attrs=["dark"])
                    for phase in phases_unformatted
                ]

        spinner = TerminalSpinner()
        spinner.writeln(HIDE_CURSOR)
        task = asyncio.ensure_future(spin(spinner))

        response = {}

        retries = 0
        max_retries = 5
        while retries <= max_retries:
            try:
                response = await openai.ChatCompletion.acreate(model=model, messages=messages, stop="-- END QUESTION")
                response_waiter.set_result(response)
                await task
                break
            except openai.error.RateLimitError as e:
                if retries == max_retries:
                    response_waiter.set_result(None)
                    await task
                    print(f"""{colored("error", "red", attrs=["bold"])}: request was rate limited - try again later.""")
                    print(f"""{colored("exception", "white", attrs=["bold"])}: {e}""")
                    sys.exit(1)

                retries += 1
                await asyncio.sleep(0.5)
                continue
            except openai.error.AuthenticationError as e:
                response_waiter.set_result(None)
                await task
                print(
                    f"""{colored("error", "red", attrs=["bold"])}: authentication error - incorrect or missing api credentials."""
                )
                print(f"""{colored("exception", "white", attrs=["bold"])}: {e}""")
                sys.exit(1)

        if response and isinstance(response, dict):
            questions = re.findall(
                r"[Qq][Uu][Ee][Ss][Tt][Ii][Oo][Nn] --(.*)",
                response.get("choices", [])[0].get("message", {}).get("content", "").strip(),
            )
            answered_questions = False
            for question in questions:
                q = question.strip()
                if q:
                    if q[-1] not in ("?", ".", "!"):
                        q += "?"
                    print(colored(f"# Question: {q}", "light_yellow"))
                    print(
                        colored("# Answer: ", "blue"),
                        end="",
                    )
                    question_answer = input().strip()
                    question_answers.append(
                        {
                            "role": "assistant",
                            "content": f"QUESTION -- {q} -- END QUESTION",
                        }
                    )
                    question_answers.append(
                        {
                            "role": "user",
                            "content": question_answer,
                        }
                    )
                    answered_questions = True
                    print("")

            if answered_questions:
                continue

            print(format_response(response.get("choices", [])[0].get("message", {}).get("content", "")).strip())
            sys.exit(0)

        print(f"""{colored("error", "red", attrs=["bold"])}: invalid response from api.""")
        print(f"""{colored("response", "white", attrs=["bold"])}: {response}""")
        sys.exit(1)


def cli_entrypoint() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
