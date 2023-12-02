from json import dump
from os import getenv
from typing import Optional

import openai
import typer
from openai import ChatCompletion, Completion
from openai.error import APIConnectionError, RateLimitError
from prompt_toolkit import PromptSession
from rich.markdown import Markdown
from rich.padding import Padding

from chatcli import ENV, ERROR, appdata_location, con
from chatcli.history import PromptHistory
from chatcli.utils import generate_prompt, get_tokens, load_log, read_prompt

MAX_TOKENS = 2048
MAX_TOKENS_V2 = 4096
MAX_TEMP = 2

openai.api_key = getenv("OPENAI_API_KEY")

if ENV == "DEBUG":
    from icecream import ic

    prompt_loc = f"{appdata_location}/prompts_test.json"
    ic(prompt_loc)
else:
    prompt_loc = f"{appdata_location}/prompts.json"


def startup():
    con.print(
        """
 ██████ ██   ██  █████  ████████  ██████ ██      ██ 
██      ██   ██ ██   ██    ██    ██      ██      ██ 
██      ███████ ███████    ██    ██      ██      ██ 
██      ██   ██ ██   ██    ██    ██      ██      ██ 
 ██████ ██   ██ ██   ██    ██     ██████ ███████ ██ 

        written by [link=https://github.com/madstone0-0]madstone0-0[/link]
    """,
        style="green",
    )


class Ask:
    def __init__(self):
        self.prompt_log = load_log(prompt_loc)
        self.session = PromptSession(
            history=PromptHistory(f"{appdata_location}/prompt_history")
        )

    def ask_v1(self, temp: float, tokens: int, model: str):
        if temp < 0 or temp > MAX_TEMP:
            con.print(f"Temperature cannot be below 0 or above {MAX_TEMP}", style=ERROR)
            raise typer.Exit(1)

        if tokens <= 0 or tokens > MAX_TOKENS:
            con.print(
                f"Max tokens cannot be below 0 or above {MAX_TOKENS}", style=ERROR
            )
            raise typer.Exit(1)

        startup()

        con.print(f"Temperature settings: {temp}\nMax Tokens: {tokens}\nModel: {model}")
        con.print(
            "Enter/Paste your prompt. Esc-Enter save it and exit or q to end the session"
        )

        while True:
            self.prompt_log = load_log(prompt_loc)
            prompt = read_prompt(self.session)

            # token_num = len(get_tokens(generate_prompt(prompt)))
            try:
                with con.status("Generating"):
                    response = Completion.create(
                        model=model,
                        prompt=generate_prompt(prompt),
                        temperature=temp,
                        max_tokens=tokens,
                    )

                output = f"""{response.choices[0].text}"""
                # con.print(f"""\n{output}\n""")
                con.print(Markdown(output))
                self.prompt_log.append(
                    {"model": model, "prompt": prompt, "response": output}
                )
                with open(prompt_loc, mode="w", encoding="utf-8") as f:
                    dump(self.prompt_log, f, indent=4, ensure_ascii=False)

            except APIConnectionError:
                con.print(
                    "Could not connect to API, please check your connection and try again",
                    style="red bold",
                )
                continue

    def ask_v2(
        self,
        temp: float,
        tokens: int,
        model: str,
        persona: str,
        is_file: Optional[bool],
    ):
        if temp < 0 or temp > MAX_TEMP:
            con.print(f"Temperature cannot be below 0 or above {MAX_TEMP}", style=ERROR)
            raise typer.Exit(1)

        if tokens <= 0 or tokens > MAX_TOKENS_V2:
            con.print(
                f"Max tokens cannot be below 0 or above {MAX_TOKENS_V2}", style=ERROR
            )
            raise typer.Exit(1)

        if is_file:
            with open(persona, "r", encoding="utf-8") as f:
                persona = "".join(f.readlines())

        startup()

        con.print(f"Temperature settings: {temp}\nMax Tokens: {tokens}\nModel: {model}")
        con.print(f"Current persona settings: {persona}")
        con.print(
            "Enter/Paste your prompt. Esc-Enter save it and exit or q to end the session"
        )

        prompts = []
        responses = []
        total_tokens = 0

        while True:
            self.prompt_log = load_log(prompt_loc)
            prompt = read_prompt(self.session)
            prompt_len = len(get_tokens(prompt))

            if prompt_len + total_tokens > tokens:
                print(
                    "You have reached the maximum token length, Please reduce the length of"
                    "the messages\nExiting...",
                    style=ERROR,
                )
                raise typer.Exit(1)

            prompts.append({"role": "user", "content": prompt})
            try:
                with con.status("Generating"):
                    response = ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": persona},
                            *prompts,
                            *responses,
                        ],
                    )

                output = response["choices"][0]["message"]["content"]
                total_tokens = response["usage"]["total_tokens"]

                responses.append({"role": "assistant", "content": output})
                con_out = Padding(
                    Markdown(f"""{output}""", code_theme="ansi_dark"), (1, 0)
                )
                con.print(con_out)
                self.prompt_log.append(
                    {
                        "model": model,
                        "persona": persona,
                        "prompt": prompt,
                        "response": output,
                    }
                )

                with open(prompt_loc, mode="w", encoding="utf-8") as f:
                    dump(self.prompt_log, f, indent=4, ensure_ascii=False)

            except RateLimitError:
                con.print(
                    "Current model currently overloaded with requests, please try again later",
                    style="red bold",
                )
                continue

            except APIConnectionError:
                con.print(
                    "Could not connect to API, please check your connection and try again",
                    style="red bold",
                )
                continue
