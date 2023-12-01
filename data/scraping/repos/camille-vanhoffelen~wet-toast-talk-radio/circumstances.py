# ruff: noqa: E501
import asyncio
import json
from pathlib import Path

import structlog
from guidance import Program
from guidance.llms import LLM

logger = structlog.get_logger()

CIRCUMSTANCES_TEMPLATE = """{{#system~}}
You are an edgy, satirical author.
{{~/system}}
{{#user~}}
Your task is to generate lists of frustrating conditions, such as mild illnesses or injuries.
Be specific about the cause of the condition.
List the condition one per line. Don't number them or print any other text, just print a condition on each line.

Here is an example:
{{#each examples~}}
{{this}}
{{/each}}

Now generate a list of {{n_circumstances}} conditions.
{{~/user}}
{{#assistant~}}
{{gen 'list' temperature=0.95 max_tokens=400}}
{{~/assistant}}"""


class Circumstances:
    """Generate lists of frustrating circumstances, such as mild illnesses or injuries, for Modern Mindfulness."""

    examples = [
        "Toe injury from excessive kink play",
        "Hungover from Tequila Tuesdays",
        "Indigestion from eating too much Vietnamese food",
    ]

    def __init__(
        self,
        llm: LLM,
        n_circumstances: int,
        n_iter: int,
        tmp_dir: Path = Path("tmp/"),
    ):
        self._llm = llm
        self.n_circumstances = n_circumstances
        self.n_iter = n_iter
        self._output_dir = tmp_dir
        logger.info(
            "Initialized circumstances", n_circumstances=n_circumstances, n_iter=n_iter
        )

    async def awrite_circumstance(self, program: Program, **kwargs):
        """For some reason the program await is messed up so we have to wrap in this async function"""
        return await program(**kwargs)

    async def awrite(self) -> bool:
        logger.info("Async writing circumstances")

        tasks = []
        for _ in range(self.n_iter):
            circumstance = Program(
                text=CIRCUMSTANCES_TEMPLATE, llm=self._llm, async_mode=True
            )
            tasks.append(
                asyncio.create_task(
                    self.awrite_circumstance(
                        circumstance,
                        examples=self.examples,
                        n_circumstances=self.n_circumstances,
                    )
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        circumstances = flatten(self.collect(results))
        circumstances += self.examples
        unique_circumstances = list(set(circumstances))
        logger.info(
            f"Generated {len(unique_circumstances)} unique circumstances",
            circumstances=unique_circumstances,
        )
        self.save(unique_circumstances)
        return True

    def collect(self, results: list[Program]) -> list[str]:
        circumstances = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("Error generating circumstances", error=r)
            else:
                circumstances.append(r["list"].split("\n"))
        return circumstances

    def save(self, circumstances: list[str]) -> None:
        with (self._output_dir / "circumstances.json").open("w") as f:
            json.dump(circumstances, f, indent=2)


def flatten(things: list) -> list:
    return [e for nested_things in things for e in nested_things]


CIRCUMSTANCES_PATH = Path(__file__).parent / "resources" / "circumstances.json"


def load_circumstances() -> list[str]:
    with CIRCUMSTANCES_PATH.open() as f:
        circumstances = json.load(f)
    return circumstances
