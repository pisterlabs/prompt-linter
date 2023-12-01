import json
import os
from datetime import datetime

import click
import openai
from art import text2art
from rich import print
from rich.live import Live
from rich.markdown import Markdown

HOME_DIR = os.path.expanduser("~")
CONFIG_PATH = os.path.join(HOME_DIR, ".v4xyz", "config.json")
CONFIG_TEMPLATE = """
{
  "openai_secret": "",
  "http_proxy": "",
  "https_proxy": ""
}
""".strip()
OPENAI_SECRET = None
HISTORY_PATH = os.path.join(HOME_DIR, ".v4xyz", "chat_history.txt")


class Cmd(click.Command):
    def format_help(self, ctx, formatter):
        art = text2art("This is the Way", "Small Slant")
        click.echo(art)
        super().format_help(ctx, formatter)


def edit_config():
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as fp:
            fp.write(CONFIG_TEMPLATE)
    click.edit(filename=CONFIG_PATH)


def load_config():
    global OPENAI_SECRET

    if not os.path.exists(CONFIG_PATH):
        click.echo("config file not found, try `v4 -e` first to set your config file.")
        exit(1)

    with open(CONFIG_PATH) as fp:
        config = json.load(fp)

    if config["http_proxy"]:
        os.environ["http_proxy"] = config["http_proxy"]
    if config["https_proxy"]:
        os.environ["https_proxy"] = config["https_proxy"]

    OPENAI_SECRET = config["openai_secret"]
    if not OPENAI_SECRET:
        click.echo("openai_secret is not set in config file, try `v4 -e` to edit your config file")
        exit(1)

    openai.api_key = OPENAI_SECRET


def count_token(content: str):
    return len(content) // 4


def load_history(max_token: int = 4096):
    histories = []
    if not os.path.exists(HISTORY_PATH):
        return [], 0, 0

    with open(HISTORY_PATH, "r") as fp:
        block_data = ""
        block_role = None
        for line in fp:
            beg_q = line.startswith("Question\t")
            beg_a = line.startswith("Answer\t")

            if beg_q or beg_a:
                last_block_data = block_data
                last_block_role = block_role
                block_data = ""
                block_role = "user" if beg_q else "assistant"
                if last_block_role is not None:
                    histories.append({"role": last_block_role, "content": last_block_data})
            else:
                block_data += line

        if block_data and block_role == "assistant":
            histories.append({"role": block_role, "content": block_data})

    total_token = 0
    histories_reverse, histories = histories[::-1], []
    for history in histories_reverse:
        # 1 token ~= 4 chars in English
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        len_content_token = count_token(history["content"])
        if total_token + len_content_token > max_token:
            break

        histories.insert(0, history)
        total_token += len_content_token

    num_dropped = len(histories_reverse) - len(histories)
    return histories, total_token, num_dropped


def ask_gpt4(inputs: str):
    histories, histories_token, num_dropped = load_history()

    system_prompt = "You are a helpful assistant always replay in markdown format."
    system_prompt += "You should answer as brief as possible."

    sys_prompt_token = count_token(system_prompt)
    input_token = count_token(inputs)
    total_token = histories_token + sys_prompt_token + input_token

    messages = [{"role": "system", "content": system_prompt}, ]
    messages.extend(histories)
    messages.append({"role": "user", "content": inputs})

    print(f"[bold red   ]Question: {inputs}[/]")
    print(f"[bold yellow]Context : {len(histories)} histories, {num_dropped} dropped, {total_token} tokens[/]")
    print("-" * 10)
    print("")

    rsp = openai.ChatCompletion.create(model="gpt-4", messages=messages, stream=True)
    answer = ""
    md = Markdown(answer)
    with Live(md, refresh_per_second=10) as live:
        for chunk in rsp:
            delta = chunk["choices"][0]["delta"]
            answer += delta.get("content", "")
            md = Markdown(answer)
            live.update(md)

    with open(HISTORY_PATH, "a") as fp:
        fp.write(f"Question\t{datetime.now()}\n{inputs}\n")

    with open(HISTORY_PATH, "a") as fp:
        fp.write(f"Answer\t{datetime.now()}\n{answer}\n")

    print("")
    print("-" * 10)
    print(f"[bold yellow]Received: {count_token(answer)} tokens[/]")


@click.command(cls=Cmd)
@click.option("-e", "--edit", is_flag=True, help="Edit the config file")
@click.argument("inputs", nargs=1, required=False)  # required must be False, otherwise we cant get into edit mode
def v4(edit, inputs):
    if edit is True:
        return edit_config()

    if inputs is None:
        click.echo(v4.get_help(click.Context(v4)))
        exit(1)

    load_config()
    inputs = inputs.strip()
    if not inputs:
        return

    inputs = inputs.capitalize()
    return ask_gpt4(inputs)


def main():
    v4()


if __name__ == "__main__":
    main()
