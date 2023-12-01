# ruff: noqa: E501
import asyncio
import json
from pathlib import Path

import structlog
from guidance import Program
from guidance.llms import LLM

logger = structlog.get_logger()

TRAIT_TEMPLATE = """{{#system~}}
You are an edgy, satirical author.
{{~/system}}
{{#user~}}
Your task is to generate lists of single-word character traits.
List the traits one per line. Don't number them or print any other text, just print a single-word trait on each line.
Here is an example:
{{#each examples~}}
{{this}}
{{/each}}
Now generate a list of {{n_traits}} traits.
{{~/user}}
{{#assistant~}}
{{gen 'list' temperature=0.95 max_tokens=200}}
{{~/assistant}}"""


class Traits:
    examples = ["Megalomania", "Cowardice", "Greed"]

    def __init__(
        self,
        llm: LLM,
        n_traits: int,
        n_iter: int,
        tmp_dir: Path = Path("tmp/"),
    ):
        self._llm = llm
        self.n_traits = n_traits
        self.n_iter = n_iter
        self._output_dir = tmp_dir
        logger.info("Initialized traits", n_traits=n_traits, n_iter=n_iter)

    async def awrite_trait(self, program: Program, **kwargs):
        """For some reason the program await is messed up so we have to wrap in this async function"""
        return await program(**kwargs)

    async def awrite(self) -> bool:
        logger.info("Async writing traits")

        tasks = []
        for _ in range(self.n_iter):
            trait = Program(text=TRAIT_TEMPLATE, llm=self._llm, async_mode=True)
            tasks.append(
                asyncio.create_task(
                    self.awrite_trait(
                        trait, examples=self.examples, n_traits=self.n_traits
                    )
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        traits = flatten(self.collect(results))
        traits += self.examples
        unique_traits = list(set(traits))
        logger.info(
            f"Generated {len(unique_traits)} unique traits", traits=unique_traits
        )
        self.save(unique_traits)
        return True

    def collect(self, results: list[Program]) -> list[str]:
        traits = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("Error generating traits", error=r)
            else:
                traits.append(r["list"].split("\n"))
        return traits

    def save(self, traits: list[str]) -> None:
        with (self._output_dir / "traits.json").open("w") as f:
            json.dump(traits, f, indent=2)


def flatten(things: list) -> list:
    return [e for nested_things in things for e in nested_things]


TRAITS_PATH = Path(__file__).parent / "resources" / "traits.json"


def load_traits() -> list[str]:
    with TRAITS_PATH.open() as f:
        traits = json.load(f)
    return traits
