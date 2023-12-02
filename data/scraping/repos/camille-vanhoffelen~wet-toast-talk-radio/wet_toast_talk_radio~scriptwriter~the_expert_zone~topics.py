import asyncio
import json
import random
from pathlib import Path

import structlog
from guidance import Program
from guidance.llms import LLM

logger = structlog.get_logger()

TOPIC_TEMPLATE = """{{#system~}}
You are an edgy, satirical author.
{{~/system}}
{{#user~}}
Your task is to generate lists of fake academic fields.
{{topic_description}}
List the fields one per line. Don't number them or print any other text, just print a field on each line.

Here is an example:
{{#each examples~}}
{{this}}
{{/each}}
Now generate a list of {{n_topics}} academic fields.
{{~/user}}
{{#assistant~}}
{{gen 'list' temperature=0.95 max_tokens=1000}}
{{~/assistant}}"""

FAKE_EXAMPLES = [
    "Nanoeconomics",
    "Oval Universe Theory",
    "Metabiology",
    "Geometric Theology",
]
FAKE_DESCRIPTION = "The academic fields should sound like real academic research fields, but be completely fake."

ABSURD_EXAMPLES = [
    "Time Travel Philosophy",
    "Plant Linguistics",
    "Cooties Virology",
    "Perpetual Motion Physics",
]
ABSURD_DESCRIPTION = (
    "The academic fields should be studies of things that are impossible and absurd."
)

MODERN_EXAMPLES = [
    "Tiktok Dance Studies",
    "Selfie Engineering",
    "Meme Anthropology",
    "Fast Fashion Economics",
]
MODERN_DESCRIPTION = (
    "The academic fields should be studies of modern phenomena that define millennial and gen-z culture. "
    "The fields should sound like real academic research fields."
)

MUNDANE_EXAMPLES = [
    "Lemonade Stand Economics",
    "Tea Infusion Chemistry",
    "History of Flatulence",
    "Dust Dynamics",
    "Yawn Physiology",
    "Handshake Mechanics",
    "Potato Aerodynamics",
]
MUNDANE_DESCRIPTION = (
    "The academic fields should be studies of common objects or mundane occurrences "
    "that people experience in their everyday lives. "
    "These should not be traditionally associated with serious research."
)


class TopicGenerator:
    def __init__(  # noqa: PLR0913
        self,
        llm: LLM,
        n_topics: int,
        n_iter: int,
        examples: list[str],
        topic_description: str,
    ):
        self._llm = llm
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.examples = examples
        self.topic_description = topic_description
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
                        topic_description=self.topic_description,
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
        self._generators = self.init_generators()
        self._output_dir = tmp_dir
        logger.info("Initialized topics", n_topics=n_topics, n_iter=n_iter)

    async def awrite(self) -> bool:
        logger.info("Async writing topics")
        tasks = []
        for g in self._generators:
            tasks.append(asyncio.create_task(g.awrite()))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        topics = flatten(results)
        self.save(topics)
        return True

    def init_generators(self) -> list[TopicGenerator]:
        fake = TopicGenerator(
            llm=self._llm,
            n_topics=self.n_topics,
            n_iter=self.n_iter,
            examples=FAKE_EXAMPLES,
            topic_description=FAKE_DESCRIPTION,
        )
        absurd = TopicGenerator(
            llm=self._llm,
            n_topics=self.n_topics,
            n_iter=self.n_iter,
            examples=ABSURD_EXAMPLES,
            topic_description=ABSURD_DESCRIPTION,
        )
        modern = TopicGenerator(
            llm=self._llm,
            n_topics=self.n_topics,
            n_iter=self.n_iter,
            examples=MODERN_EXAMPLES,
            topic_description=MODERN_DESCRIPTION,
        )
        mundane = TopicGenerator(
            llm=self._llm,
            n_topics=self.n_topics,
            n_iter=self.n_iter,
            examples=MUNDANE_EXAMPLES,
            topic_description=MUNDANE_DESCRIPTION,
        )
        return [fake, absurd, modern, mundane]

    def save(self, topics: list[str]) -> None:
        with (self._output_dir / "the-expert-zone-topics.json").open("w") as f:
            json.dump(topics, f, indent=2)


def flatten(things: list) -> list:
    return [e for nested_things in things for e in nested_things]


TOPICS_PATH = Path(__file__).parent / "resources" / "the-expert-zone-topics.json"
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
