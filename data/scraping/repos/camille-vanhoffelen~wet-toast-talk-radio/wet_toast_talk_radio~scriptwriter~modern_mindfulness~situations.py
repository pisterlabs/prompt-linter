# ruff: noqa: E501
import asyncio
import json
from asyncio import Task
from pathlib import Path

import structlog
from guidance import Program
from guidance.llms import LLM

logger = structlog.get_logger()

SITUATIONS_TEMPLATE = """{{#system~}}
{{writer}}
{{~/system}}
{{#user~}}
Your task is to generate lists of {{description}}. Be specific.
List the situations one per line. Don't number them or print any other text, just print a situation on each line.

Here is an example:
{{#each examples~}}
{{this}}
{{/each}}

Now generate a list of {{n_situations}} situations.
{{~/user}}
{{#assistant~}}
{{gen 'list' temperature=0.95 max_tokens=400}}
{{~/assistant}}"""

MUNDANE_WRITER = "You are an edgy, satirical writer."
MUNDANE_EXAMPLES = [
    "going to the supermarket to buy frozen peas",
    "going to pick up a package from the post office",
    "cooking a meal for the family",
]
MUNDANE_DESCRIPTION = "mundane and harmless daily situations"
RARE_WRITER = "You are a writer."
RARE_EXAMPLES = [
    "going to a concert of your favorite band",
    "celebrating your 5th anniversary dinner with your spouse",
    "attending an interview to your dream job",
    "going skydiving for the first time",
]
RARE_DESCRIPTION = "exceptional events that you look forward to"


class Situations:
    """Generate lists of contextual situations for Modern Mindfulness."""

    def __init__(
        self,
        llm: LLM,
        n_situations: int,
        n_iter: int,
        tmp_dir: Path = Path("tmp/"),
    ):
        self._llm = llm
        self.n_situations = n_situations
        self.n_iter = n_iter
        self._output_dir = tmp_dir
        logger.info("Initialized situations", n_situations=n_situations, n_iter=n_iter)

    async def awrite_situation(self, program: Program, **kwargs):
        """For some reason the program await is messed up so we have to wrap in this async function"""
        return await program(**kwargs)

    async def awrite(self) -> bool:
        logger.info("Async writing situations")
        rare_tasks = self.make_tasks(
            description=RARE_DESCRIPTION, examples=RARE_EXAMPLES, writer=RARE_WRITER
        )
        mundane_tasks = self.make_tasks(
            description=MUNDANE_DESCRIPTION,
            examples=MUNDANE_EXAMPLES,
            writer=MUNDANE_WRITER,
        )
        tasks = rare_tasks + mundane_tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        situations = flatten(self.collect(results))
        situations += RARE_EXAMPLES
        situations += MUNDANE_EXAMPLES
        unique_situations = list(set(situations))
        logger.info(
            f"Generated {len(unique_situations)} unique situations",
            situations=unique_situations,
        )
        self.save(unique_situations)
        return True

    def make_tasks(
        self,
        description: str,
        examples: list[str],
        writer: str,
    ) -> list[Task]:
        tasks = []
        for _ in range(self.n_iter):
            situation = Program(
                text=SITUATIONS_TEMPLATE, llm=self._llm, async_mode=True
            )
            tasks.append(
                asyncio.create_task(
                    self.awrite_situation(
                        situation,
                        writer=writer,
                        examples=examples,
                        description=description,
                        n_situations=self.n_situations,
                    )
                )
            )
        return tasks

    def collect(self, results: list[Program]) -> list[str]:
        situations = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("Error generating situations", error=r)
            else:
                situations.append(r["list"].split("\n"))
        return situations

    def save(self, situations: list[str]) -> None:
        with (self._output_dir / "situations.json").open("w") as f:
            json.dump(situations, f, indent=2)


def flatten(things: list) -> list:
    return [e for nested_things in things for e in nested_things]


SITUATIONS_PATH = Path(__file__).parent / "resources" / "situations.json"


def load_situations() -> list[str]:
    with SITUATIONS_PATH.open() as f:
        situations = json.load(f)
    return situations
