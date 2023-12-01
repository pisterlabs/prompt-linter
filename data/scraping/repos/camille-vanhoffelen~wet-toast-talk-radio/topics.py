# ruff: noqa: E501
import asyncio
import json
import random
from pathlib import Path

import structlog
from guidance import Program
from guidance.llms import LLM

logger = structlog.get_logger()

EXAMPLES = [
    "I'm having trouble with my partner.",
    "I don't enjoy sex anymore.",
    "How do I know if I'm in love?",
    "How do I know my sexuality?",
    "How do I break up with my partner?",
    "I have feelings for my best friend.",
    "I have feelings for two people at once.",
    "I keep getting ghosted... What am I doing wrong?",
    "How do I ask my crush out?",
]

TOPIC_TEMPLATE = """{{#system~}}
You are an expert in dating and relationships.
{{~/system}}
{{#user~}}
Your task is to generate lists of issues and questions that people are likely to ask you about love, sex, dating, and relationships.
List the fields one per line. Don't number them or print any other text, just print a field on each line.

Here is an example:
{{#each examples~}}
{{this}}
{{/each}}

Now generate a list of {{n_topics}} issues and questions.
{{~/user}}
{{#assistant~}}
{{gen 'list' temperature=1.5 max_tokens=1000}}
{{~/assistant}}"""


class TopicGenerator:
    def __init__(
        self,
        llm: LLM,
        n_topics: int,
        n_iter: int,
        examples: list[str],
    ):
        self._llm = llm
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.examples = examples
        logger.info("Initialized topics", n_topics=n_topics, n_iter=n_iter)

    async def awrite_topic(self, program: Program, **kwargs) -> Program:
        """For some reason the program await is messed up so we have to wrap in this async function"""
        return await program(**kwargs)

    async def awrite(self) -> list[str]:
        logger.info("Async writing topics")

        tasks = []
        for _ in range(self.n_iter):
            topic = Program(text=TOPIC_TEMPLATE, llm=self._llm, async_mode=True)
            tasks.append(
                asyncio.create_task(
                    self.awrite_topic(
                        topic,
                        examples=self.examples,
                        n_topics=self.n_topics,
                    )
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        topics = self.collect(results)
        topics += self.examples
        unique_topics = list(set(topics))
        logger.info(
            f"Generated {len(unique_topics)} unique topics", topics=unique_topics
        )
        return unique_topics

    def collect(self, results: list[Program]) -> list[str]:
        all_topics = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("Error generating topics", error=r)
            else:
                logger.info(r)
                topics = r["list"].split("\n")
                topics = [t.strip().replace("-", " ") for t in topics]
                all_topics.extend(topics)
        return all_topics


class Topics:
    def __init__(
        self,
        llm: LLM,
        n_topics: int,
        n_iter: int,
        tmp_dir: Path = Path("tmp/"),
    ):
        self._llm = llm
        self.n_topics = n_topics
        self.n_iter = n_iter
        self._generator = TopicGenerator(
            llm=self._llm,
            n_topics=self.n_topics,
            n_iter=self.n_iter,
            examples=EXAMPLES,
        )
        self._output_dir = tmp_dir
        logger.info("Initialized topics", n_topics=n_topics, n_iter=n_iter)

    async def awrite(self) -> bool:
        logger.info("Async writing topics")
        results = await self._generator.awrite()
        self.save(results)
        return True

    def save(self, topics: list[str]) -> None:
        with (self._output_dir / "prolove-topics.json").open("w") as f:
            json.dump(topics, f, indent=2)


TOPICS_PATH = Path(__file__).parent / "resources" / "prolove-topics.json"
TOPICS_CACHE = None


def load_topics() -> list[str]:
    global TOPICS_CACHE  # noqa: PLW0603
    if TOPICS_CACHE is None:
        with TOPICS_PATH.open() as f:
            TOPICS_CACHE = json.load(f)
    return TOPICS_CACHE


def random_topic() -> str:
    """Return a random topic from the list of topics"""
    return random.choice(load_topics())
