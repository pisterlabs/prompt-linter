#!/usr/bin/env python3
import os
from pathlib import Path
import click
import openai
import sys
from ruamel.yaml import YAML

openai.organization = os.getenv("OPENAI_ORG_ID")
openai.api_key = os.getenv("OPENAI_API_KEY")


# Define the command line interface using Click
@click.command(context_settings=dict(show_default=True))
@click.argument("yamlfile", nargs=1, required=True, type=str)
@click.option(
    "-m",
    "--model",
    default="gpt-4",
    help="The UUID of the model to use.",
    show_default=True,
)
@click.option(
    "-T",
    "--temperature",
    default=1.0,
    help="The sampling temperature of the model. Higher values means the model is more bold (creative).",
    type=float,
    show_default=True,
)
@click.option(
    "-p",
    "--top_p",
    default=1.0,
    type=float,
    show_default=True,
    help="The probability of sampling the most likely (top) token.",
)
@click.option(
    "-N",
    "--max-tokens",
    default=2000,
    help="The maximum number of tokens in the generated message.",
    type=int,
    show_default=True,
)
@click.option(
    "-P",
    "--presence-penalty",
    default=0.0,
    type=float,
    help="The presence penalty coefficient. This punishes new tokens based on whether they already appear in the context so far. Set this to 0 to disable.",
    show_default=True,
)
@click.option(
    "-F",
    "--frequency-penalty",
    default=0.0,
    type=float,
    help="The frequency penalty coefficient. This punishes new tokens based on their existing frequency in the response text so far. Set this to 0 to disable.",
    show_default=True,
)
@click.option(
    "-u",
    "--user",
    default="testuser",
    type=str,
    show_default=True,
    help="The identifier metadata to associate with the message.",
)
@click.option(
    "-s",
    "--save",
    default=None,
    type=click.Path(path_type=Path),
    help="The path to save the resulting chat history. Optional.",
)

# Define the main function and add type hints
def main(
    yamlfile:str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    presence_penalty: float,
    frequency_penalty: float,
    user: str,
    save: Path | None,
) -> None:

    yamlmessages = YAML().load(Path(yamlfile))

    messages = []
    for idx, message in enumerate(yamlmessages):
        role, msg = tuple(message.items())[0]
        if role not in ("system", "user", "userfile", "assistant"):
            raise NotImplementedError("Role not implemented", role)
        if role == "userfile":
            content = Path(msg).read_file()
            msg = f"```\n{content}```"
            role = "user"

        messages.append({"role": role, "content": msg.strip()})

    completion = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        stream=True,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        user=user,
    )

    result = ""

    for part in completion:
        if "content" in part["choices"][0]["delta"]:
            sys.stdout.write(x := part["choices"][0]["delta"]["content"])
            sys.stdout.flush()
            result = result + x
    print()

if __name__ == "__main__":
    main()
