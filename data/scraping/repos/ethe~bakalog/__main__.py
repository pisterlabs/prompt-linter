import logging
import os
from glob import glob

import click
import openai
from IPython import embed
from rich.logging import RichHandler

from bakalog import Match, Sink, collect
from bakalog.cluster import Cluster
from bakalog.extract import extract
from bakalog.util import Memory, parse_size


@click.group()
def main():
    pass


@main.command(
    help="bakalog cache all extracted patterns to each files as default, clean the cache as needed."
)
def clean():
    files = glob(f"{Memory.PATH}/*")
    for f in files:
        os.remove(f)


@main.command()
@click.argument("file")
@click.option(
    "--gpt-base",
    default=openai.api_base,
    help="OpenAI API base.",
    show_default=True,
)
@click.option(
    "--max-lines",
    default=4096,
    help="Max log lines would be parsed, set 0 to parese all logs.",
    type=int,
    show_default=True,
)
@click.option(
    "--buf-size",
    default="2MB",
    help="Number of logs to cluster detection.",
    show_default=True,
)
@click.option(
    "--max-len",
    default=512,
    help="Max length of each log, rest of log would be dropped.",
    type=int,
    show_default=True,
)
@click.option(
    "--threshold",
    default=0.85,
    help="Threshold of logs clustering.",
    type=float,
    show_default=True,
)
def run(file, gpt_base, max_lines, buf_size, max_len, threshold):
    if "OPENAI_API_KEY" not in os.environ:
        logging.error(
            "the tool relies on GPT4, please set env: `OPENAI_API_KEY` as OpenAI API key."
        )
        return

    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    buf_size = parse_size(buf_size)

    with Memory().current(file):
        f = Sink(file, max_size=max_len)
        m = Match(Memory(), f)
        c = Cluster(f, m, buf_size=buf_size, threshold=threshold)
        e = extract(
            c,
            m,
            api_base=gpt_base,
            model="gpt-4",
            temperature=0,
        )
        result = collect(e, max_lines)

        embed(header="use variable `result` to get the result")


if __name__ == "__main__":
    main()
