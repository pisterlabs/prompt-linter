import asyncio
import json
import random
from pathlib import Path

import structlog
from guidance import Program
from guidance.llms import LLM

logger = structlog.get_logger()

DATING_EXAMPLES = [
    # "I left a date because they weren't impressed by my sudoku high scores.",
    # "I ghosted someone I had been dating for months because they changed perfume.",
    "I went on a date with someone who only communicated through interpretive dance.",
    "I went on a date with someone who brought their pet tarantula along in their purse.",
]

TOPIC_TEMPLATE = """{{#system~}}
You are an edgy, satirical writer.
{{~/system}}
{{#user~}}
Your task is to generate lists of weird and unusual dating experiences.
The experiences must be specific and personal.
List the experiences one per line. Don't number them or print any other text, just print a field on each line.

Here is an example:
{{#each examples~}}
{{this}}
{{/each}}

Now generate a list of {{n_anecdotes}} dating experiences.
{{~/user}}
{{#assistant~}}
{{gen 'list' temperature=1.0 max_tokens=1000}}
{{~/assistant}}"""


class AnecdoteGenerator:
    def __init__(
        self,
        llm: LLM,
        n_anecdotes: int,
        n_iter: int,
        examples: list[str],
    ):
        self._llm = llm
        self.n_anecdotes = n_anecdotes
        self.n_iter = n_iter
        self.examples = examples
        logger.info("Initialized anecdotes", n_anecdotes=n_anecdotes, n_iter=n_iter)

    async def awrite_anecdote(self, program: Program, **kwargs) -> Program:
        """For some reason the program await is messed up so we have to wrap in this async function"""
        return await program(**kwargs)

    async def awrite(self) -> list[str]:
        logger.info("Async writing anecdotes")

        tasks = []
        for _ in range(self.n_iter):
            anecdote = Program(text=TOPIC_TEMPLATE, llm=self._llm, async_mode=True)
            tasks.append(
                asyncio.create_task(
                    self.awrite_anecdote(
                        anecdote,
                        examples=self.examples,
                        n_anecdotes=self.n_anecdotes,
                    )
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        anecdotes = self.collect(results)
        anecdotes += self.examples
        unique_anecdotes = list(set(anecdotes))
        logger.info(
            f"Generated {len(unique_anecdotes)} unique anecdotes",
            anecdotes=unique_anecdotes,
        )
        return unique_anecdotes

    def collect(self, results: list[Program]) -> list[str]:
        all_anecdotes = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("Error generating anecdotes", error=r)
            else:
                logger.info(r)
                anecdotes = r["list"].split("\n")
                anecdotes = [t.strip().replace("-", " ") for t in anecdotes]
                all_anecdotes.extend(anecdotes)
        return all_anecdotes


class Anecdotes:
    def __init__(
        self,
        llm: LLM,
        n_anecdotes: int,
        n_iter: int,
        tmp_dir: Path = Path("tmp/"),
    ):
        self._llm = llm
        self.n_anecdotes = n_anecdotes
        self.n_iter = n_iter
        self._generators = self.init_generators()
        self._output_dir = tmp_dir
        logger.info("Initialized anecdotes", n_anecdotes=n_anecdotes, n_iter=n_iter)

    async def awrite(self) -> bool:
        logger.info("Async writing anecdotes")
        tasks = []
        for g in self._generators:
            tasks.append(asyncio.create_task(g.awrite()))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        topics = flatten(results)
        self.save(topics)
        return True

    def init_generators(self) -> list[AnecdoteGenerator]:
        dating = AnecdoteGenerator(
            llm=self._llm,
            n_anecdotes=self.n_anecdotes,
            n_iter=self.n_iter,
            examples=DATING_EXAMPLES,
        )
        return [dating]

    def save(self, anecdotes: list[str]) -> None:
        with (self._output_dir / "prolove-anecdotes.json").open("w") as f:
            json.dump(anecdotes, f, indent=2)


def flatten(things: list) -> list:
    return [e for nested_things in things for e in nested_things]


TOPICS_PATH = Path(__file__).parent / "resources" / "prolove-anecdotes.json"
TOPICS_CACHE = None


def load_anecdotes() -> list[str]:
    global TOPICS_CACHE  # noqa: PLW0603
    if TOPICS_CACHE is None:
        with TOPICS_PATH.open() as f:
            TOPICS_CACHE = json.load(f)
    return TOPICS_CACHE


def random_anecdote() -> str:
    """Return a random anecdote from the list of anecdotes"""
    return random.choice(load_anecdotes())
